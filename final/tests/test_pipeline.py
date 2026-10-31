import json
import struct
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import sys
import cv2
import numpy as np
from scanner.io import read_ply,write_ply
from scanner.geometry import ray_first_hit,platform_pose
from scanner.reconstruct import read_model,reconstruct,resume_dense
from scanner.demo import make_demo
from scanner.quality import analyze
from scanner.planner import recommend
from scanner.dense_masks import write_stereo_array,read_stereo_array


class PipelineTests(unittest.TestCase):
    def test_binary_ply_normal_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'binary.ply'
            header='ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n'
            with p.open('wb') as f:
                f.write(header.encode())
                for v in [(0,0,0),(1,0,0),(0,1,0)]: f.write(struct.pack('<6f',*v,0,0,1))
                f.write(struct.pack('<B3i',3,0,1,2))
            vertices,faces,attrs=read_ply(p,True)
            self.assertEqual(faces.tolist(),[[0,1,2]])
            np.testing.assert_allclose(attrs['nz'],1)
            np.testing.assert_allclose(vertices[1],[1,0,0])

    def test_occluding_surface_wins(self):
        vertices=np.array([[-2,-2,1],[2,-2,1],[0,2,1],[-2,-2,2],[2,-2,2],[0,2,2]],float)
        distances=ray_first_hit(np.array([[0,0,0],[5,0,0]],float),np.array([[0,0,1],[0,0,1]],float),vertices,np.array([[0,1,2],[3,4,5]]))
        self.assertAlmostEqual(distances[0],1)
        self.assertTrue(np.isinf(distances[1]))

    def test_colmap_pose_and_empty_observation_line(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'cameras.txt').write_text('1 SIMPLE_RADIAL 800 600 650 400 300 0.01\n')
            (root/'images.txt').write_text('# comment\n1 1 0 0 0 0 0 3 1 phoneA/frame one.jpg\n\n2 1 0 0 0 1 0 3 1 phoneA/two.jpg\n0 0 -1\n')
            cameras=read_model(root)
            self.assertEqual(len(cameras),2)
            self.assertEqual(cameras[0]['file'],'phoneA/frame one.jpg')
            np.testing.assert_allclose(cameras[0]['center'],[0,0,-3])
            self.assertAlmostEqual(cameras[0]['K'][0][2],399.5)

    def test_rotations_about_pivots(self):
        cfg={'yaw_axis':[0,0,1],'yaw_pivot':[0,0,0],'pitch_axis':[1,0,0],
             'pitch_pivot':[0,0,1],'platform_from_model':np.eye(4).tolist()}
        T=platform_pose(cfg,0,90)
        np.testing.assert_allclose(T@np.array([0,0,1,1]),[0,0,1,1],atol=1e-8)
        np.testing.assert_allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-8)

    def test_demo_evidence_and_safe_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); mesh,cloud,cameras,cfg=make_demo(root)
            report=analyze(mesh,cloud,cameras,root/'quality',80)
            self.assertGreater(len(report['boundary_components']),0)
            self.assertGreater(report['unsupported_fraction'],0)
            result=recommend(mesh,report,cfg,root/'plan')
            self.assertFalse(result['executable'])
            self.assertTrue(result['recommended'])
            self.assertTrue(all(-45<=p['pitch']<=45 for p in result['recommended']))
            self.assertIsNone(report['samples'][0]['fixture_blocked_views'])

    def test_reconstruction_orchestration_preserves_fused_normals(self):
        """Mock COLMAP processes: checks interfaces, not real SfM/MVS accuracy."""
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); images=root/'images'; masks=root/'masks'
            (images/'phoneA').mkdir(parents=True); (masks/'phoneA').mkdir(parents=True)
            for i in range(3):
                cv2.imwrite(str(images/'phoneA'/f'{i}.jpg'),np.full((64,64),127,np.uint8))
                mask=np.zeros((64,64),np.uint8); mask[8:56,8:56]=255
                cv2.imwrite(str(masks/'phoneA'/f'{i}.jpg.png'),mask)
            points=np.array([[i*.005,j*.005,0] for i in range(5) for j in range(5)])
            normals=np.tile([0,0,1],(len(points),1))
            meshing_calls=[]
            stereo_calls=[]
            def fake_run(cmd,**kwargs):
                command=cmd[1]
                if command=='patch_match_stereo': stereo_calls.append(cmd)
                if '-h' in cmd:
                    return SimpleNamespace(returncode=0,stdout='--FeatureExtraction.use_gpu --FeatureMatching.use_gpu',stderr='')
                def argument(flag): return Path(cmd[cmd.index(flag)+1])
                if command=='image_undistorter':
                    out=argument('--output_path')
                    (out/'images/phoneA').mkdir(parents=True,exist_ok=True)
                    for i in range(3):
                        cv2.imwrite(str(out/'images/phoneA'/f'{i}.jpg'),np.full((64,64,3),127,np.uint8))
                if command=='patch_match_stereo':
                    out=argument('--workspace_path')
                    kind='geometric' if cmd[cmd.index('--PatchMatchStereo.geom_consistency')+1]=='true' else 'photometric'
                    if kind=='geometric':
                        for i in range(3):
                            previous=read_stereo_array(out/'stereo/depth_maps/phoneA'/f'{i}.jpg.photometric.bin')
                            self.assertTrue(np.all(previous[:8]==0))
                            self.assertEqual(previous[32,32,0],1)
                    for directory,channels in [('depth_maps',1),('normal_maps',3)]:
                        folder=out/'stereo'/directory/'phoneA'; folder.mkdir(parents=True,exist_ok=True)
                        for i in range(3):
                            write_stereo_array(folder/f'{i}.jpg.{kind}.bin',np.ones((64,64,channels),np.float32))
                if command=='mapper':
                    model=argument('--output_path')/'0'; model.mkdir(); (model/'images.bin').write_bytes(b'mock')
                if command=='model_converter':
                    out=argument('--output_path')
                    if cmd[cmd.index('--output_type')+1]=='TXT':
                        (out/'cameras.txt').write_text('1 SIMPLE_RADIAL 64 64 50 32 32 0\n')
                        (out/'images.txt').write_text(''.join(f'{i+1} 1 0 0 0 0 0 3 1 phoneA/{i}.jpg\n\n' for i in range(3)))
                        (out/'points3D.txt').write_text('1 0 0 0 100 100 100 0.25 1 0 2 0\n')
                    else: write_ply(out,points)
                if command=='stereo_fusion':
                    out=argument('--workspace_path')
                    for i in range(3):
                        depth=read_stereo_array(out/'stereo/depth_maps/phoneA'/f'{i}.jpg.geometric.bin')
                        self.assertTrue(np.all(depth[:8]==0))
                        self.assertEqual(depth[32,32,0],1)
                    write_ply(argument('--output_path'),points,normals=normals)
                if command=='poisson_mesher':
                    meshing_calls.append(cmd)
                    _,_,attributes=read_ply(argument('--input_path'),True)
                    np.testing.assert_allclose(attributes['nz'],1)
                    if '--PoissonMeshing.trim' in cmd:
                        write_ply(argument('--output_path'),points,[[0,1,5]])
                    else:
                        write_ply(argument('--output_path'),np.empty((0,3)))
                return SimpleNamespace(returncode=0)
            with patch('scanner.reconstruct.subprocess.run',side_effect=fake_run):
                cameras=reconstruct(images,root/'scan',sys.executable,masks,patch_window_radius=10)
            self.assertEqual(len(cameras),3)
            self.assertTrue(json.loads((root/'scan/reconstruction_status.json').read_text())['dense_complete'])
            self.assertTrue((root/'scan/sparse_statistics.json').is_file())
            self.assertEqual(len(meshing_calls),2)
            self.assertIn('--PoissonMeshing.trim',meshing_calls[-1])
            self.assertEqual(json.loads((root/'scan/reconstruction_status.json').read_text())['mesh_faces'],1)
            self.assertEqual(stereo_calls[0][stereo_calls[0].index('--PatchMatchStereo.window_radius')+1],'10')
            status_path=root/'scan/reconstruction_status.json'
            status=json.loads(status_path.read_text())
            status.update(dense_complete=False,chosen_model=str(root/'scan/sparse/0'))
            status_path.write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError,'same masks'):
                resume_dense(images,root/'scan',sys.executable,root/'other_masks')
            resumed=[]
            def fake_resume(cmd,**kwargs):
                self.assertNotIn(cmd[1],['feature_extractor','exhaustive_matcher','mapper'])
                resumed.append(cmd)
                return fake_run(cmd,**kwargs)
            with patch('scanner.reconstruct.subprocess.run',side_effect=fake_resume):
                resumed_cameras=resume_dense(images,root/'scan',sys.executable,masks,patch_window_radius=10)
            self.assertEqual(len(resumed_cameras),3)
            self.assertEqual(resumed[0][1],'image_undistorter')
            self.assertEqual(resumed[0][resumed[0].index('--input_path')+1],str(root/'scan/sparse/0'))
            self.assertTrue(json.loads(status_path.read_text())['dense_complete'])
            self.assertEqual(stereo_calls[-1][stereo_calls[-1].index('--PatchMatchStereo.window_radius')+1],'10')


if __name__=='__main__': unittest.main()
