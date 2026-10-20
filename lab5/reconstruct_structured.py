"""Dense stereo from captured Gray-code and four-step phase sequences.

Uses both projected axes as temporal correspondence signatures. Missing high
Gray bits mean codes repeat; calibrated epipolar geometry resolves candidates.
Scene pose is estimated from static appearance; baseline scale is inherited.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lab4'))
from assignment4 import ply, cameras

KEPT = [i for i in range(1,31) if i not in (2,11)]


def read(path):
    im=cv.imdecode(np.fromfile(path,np.uint8),cv.IMREAD_COLOR)
    if im is None:
        raise ValueError(path)
    return im


def picture(path,im):
    cv.imencode(path.suffix,im)[1].tofile(str(path))


def gray_preview(depth,mask,limits=None):
    out=np.zeros(mask.shape,np.uint8)
    if mask.any():
        low,high=np.percentile(depth[mask],[2,98]) if limits is None else limits
        out[mask]=np.rint(255-254*np.clip((depth[mask]-low)/max(high-low,1e-6),0,1)).astype(np.uint8)
    return out


def signature(stack):
    descriptors=[]; contrasts=[]; amplitudes=[]
    for offset in (0,14):
        for j in range(0,10,2):
            delta=stack[offset+j]-stack[offset+j+1]
            descriptors.append(np.where(delta>=0,.5,-.5))
            contrasts.append(np.abs(delta))
        a,b,c,d=stack[offset+10:offset+14]
        cosine=a-c; sine=d-b
        length=np.sqrt(cosine*cosine+sine*sine)
        descriptors.extend([cosine/np.maximum(length,1),sine/np.maximum(length,1)])
        amplitudes.append(length*.5)
    descriptor=np.stack(descriptors,axis=-1).astype(np.float32)
    modulation=np.minimum(amplitudes[0],amplitudes[1])
    mask=(modulation>=4.) & (np.min(contrasts,axis=0)>=1.5)
    return descriptor,mask,modulation


def dense_matches(left,right,left_valid,right_valid,max_disparity):
    ly,lx=np.nonzero(left_valid);ry,rx=np.nonzero(right_valid)
    print('Modulated/confident pixels:',len(lx),len(rx),flush=True)
    if min(len(lx),len(rx))<100:
        raise ValueError('Too few pixels with reliable temporal modulation')
    cv.setRNGSeed(42)
    index=cv.flann_Index(right[ry,rx],dict(algorithm=1,trees=6))
    disparity=np.full(left_valid.shape,np.nan,np.float32)
    error=np.full(left_valid.shape,np.nan,np.float32)
    row_error=np.full(left_valid.shape,np.nan,np.float32)
    candidate_y=np.full(left_valid.shape,-1,np.int32)
    candidate_x=np.full(left_valid.shape,-1,np.int32)
    for start in range(0,len(lx),6000):
        y,x=ly[start:start+6000],lx[start:start+6000]
        ids,dist=index.knnSearch(left[y,x],min(64,len(rx)),params={'checks':96})
        xx,yy=rx[ids],ry[ids]
        delta=x[:,None]-xx
        eligible=(np.abs(y[:,None]-yy)<=2)&(delta>1)&(delta<max_disparity)&(dist<.55)
        # Prefer close temporal signatures; y distance resolves near ties only.
        score=np.where(eligible,dist+.015*np.abs(y[:,None]-yy),np.inf)
        best=np.argmin(score,axis=1)
        rows=np.arange(len(x));ok=np.isfinite(score[rows,best])
        disparity[y[ok],x[ok]]=delta[rows,best][ok]
        error[y[ok],x[ok]]=dist[rows,best][ok]
        row_error[y[ok],x[ok]]=np.abs(y[:,None]-yy)[rows,best][ok]
        candidate_y[y[ok],x[ok]]=yy[rows,best][ok]
        candidate_x[y[ok],x[ok]]=xx[rows,best][ok]
    return disparity,error,row_error,candidate_y,candidate_x


def projector_coordinates(descriptor):
    coordinates=[]
    lut=np.array([int(i).bit_count() for i in range(32)],np.uint8)
    for base in (0,7):
        code=np.zeros(descriptor.shape[:2],np.int32)
        for j in range(5):
            code|=(descriptor[...,base+j]>0).astype(np.int32)<<(4-j)
        phase=np.mod(np.arctan2(descriptor[...,base+6],descriptor[...,base+5]),2*np.pi)*128/(2*np.pi)
        options=np.stack((phase,phase+128),axis=-1)
        integer=np.rint(options).astype(np.int32)%256
        expected=((integer^(integer>>1))>>2)&31
        penalty=lut[expected^code[...,None]]
        chosen=np.argmin(penalty,axis=-1)
        coordinates.append(np.take_along_axis(options,chosen[...,None],axis=-1)[...,0])
    return np.stack(coordinates,axis=-1)


def local_2d_search(source,target,source_valid,target_valid,seed_x,seed_valid,radius,horizontal=6):
    """Find actual measured target pixels near a same-row stereo seed."""
    sy,sx=np.nonzero(source_valid&seed_valid)
    start_x=seed_x[sy,sx]
    best=np.full(len(sx),np.inf,np.float32)
    bx=np.full(len(sx),-1,np.int32);by=bx.copy()
    h,w=source_valid.shape
    reference=source[sy,sx]
    for dy in range(-radius,radius+1):
        ty=sy+dy;cy=np.clip(ty,0,h-1)
        for dx in range(-horizontal,horizontal+1):
            tx=start_x+dx;cx=np.clip(tx,0,w-1)
            difference=np.abs(reference-target[cy,cx])
            difference=np.minimum(difference,256-difference)
            squared=np.sum(difference*difference,axis=1)
            valid=(ty>=0)&(ty<h)&(tx>=0)&(tx<w)&target_valid[cy,cx]
            # The position terms break close phase ties, not create matches.
            score=np.where(valid,squared+.025*dx*dx+.002*dy*dy,np.inf)
            take=score<best
            best[take]=score[take];bx[take]=tx[take];by[take]=ty[take]
        if (dy+radius)%20==0:
            print(f'2D search row offset {dy:+d}/{radius}, query pixels {len(sx)}',flush=True)
    target_x=np.full(source_valid.shape,-1,np.int32);target_y=target_x.copy()
    cost=np.full(source_valid.shape,np.inf,np.float32)
    ok=np.isfinite(best)
    target_x[sy[ok],sx[ok]]=bx[ok];target_y[sy[ok],sx[ok]]=by[ok]
    delta=np.abs(reference[ok]-target[by[ok],bx[ok]])
    delta=np.minimum(delta,256-delta)
    cost[sy[ok],sx[ok]]=np.max(delta,axis=1)
    return target_x,target_y,cost


def nearest_prior(values,valid):
    """Fill a search preference, never the exported disparity/depth."""
    if not valid.any():
        return None
    _,labels=cv.distanceTransformWithLabels((~valid).astype(np.uint8),cv.DIST_L2,5,
                                           labelType=cv.DIST_LABEL_PIXEL)
    lookup=np.zeros(int(labels.max())+1,np.float32)
    lookup[labels[valid]]=values[valid]
    return lookup[labels]


def periodic_direction(source,target,source_valid,target_valid,radius,direction,prior=None):
    """Global candidates on phase circles; no same-row seed is required."""
    sy,sx=np.nonzero(source_valid);ty,tx=np.nonzero(target_valid)
    # Raw phase cos/sin are continuous across 2pi. Gray bits disambiguate cycles,
    # but have low weight so one uncertain bit cannot cause a large false jump.
    weights=np.array([.18]*5+[1.,1.]+[.18]*5+[1.,1.],np.float32)
    data=np.ascontiguousarray(target[ty,tx]*weights)
    cv.setRNGSeed(42)
    index=cv.flann_Index(data,dict(algorithm=1,trees=6))
    target_x=np.full(source_valid.shape,-1,np.int32);target_y=target_x.copy()
    errors=np.full(source_valid.shape,np.inf,np.float32)
    for start in range(0,len(sx),5000):
        y,x=sy[start:start+5000],sx[start:start+5000]
        query=source[y,x]
        ids,dist=index.knnSearch(np.ascontiguousarray(query*weights),min(160,len(tx)),params={'checks':160})
        xx,yy=tx[ids],ty[ids]
        disparity=direction*(x[:,None]-xx)
        a=query[:,None,:];b=target[yy,xx]
        # Independent circular distances for each projected axis, in pixels
        # of the 128-pixel sinusoidal period. Never subtract wrapped scalars.
        phase_errors=[]
        for base in (5,12):
            cross=a[...,base]*b[...,base+1]-a[...,base+1]*b[...,base]
            dot=a[...,base]*b[...,base]+a[...,base+1]*b[...,base+1]
            phase_errors.append(np.abs(np.arctan2(cross,dot))*128/(2*np.pi))
        phase_error=np.maximum(phase_errors[0],phase_errors[1])
        bit_mismatch=np.sum(a[...,[0,1,2,3,4,7,8,9,10,11]]!=b[...,[0,1,2,3,4,7,8,9,10,11]],axis=-1)
        eligible=(np.abs(yy-y[:,None])<=radius)&(disparity>1)&(disparity<768)
        eligible &= (phase_error<=4)&(bit_mismatch<=2)
        score=dist+.000005*(yy-y[:,None])**2
        if prior is not None:
            score += .000001*np.minimum((disparity-prior[y,x,None])**2,40000)
        score=np.where(eligible,score,np.inf)
        rows=np.arange(len(x));chosen=np.argmin(score,axis=1)
        good=np.isfinite(score[rows,chosen])
        target_x[y[good],x[good]]=xx[rows,chosen][good]
        target_y[y[good],x[good]]=yy[rows,chosen][good]
        errors[y[good],x[good]]=phase_error[rows,chosen][good]
        if start%100000==0:
            print(f'Periodic direction {direction:+}: {start}/{len(sx)}',flush=True)
    return target_x,target_y,errors


def periodic_matches(left,right,left_valid,right_valid,radius,prior_dir):
    forward_prior=reverse_prior=None
    if prior_dir is not None and (prior_dir/'valid_mask.npy').is_file():
        mask=np.load(prior_dir/'valid_mask.npy');d=np.load(prior_dir/'disparity.npy')
        if mask.shape!=left_valid.shape:
            raise ValueError('Prior and current processing dimensions differ')
        forward_prior=nearest_prior(d,mask)
        oldx=np.load(prior_dir/'matched_right_x.npy');oldy=np.load(prior_dir/'matched_right_y.npy')
        sy,sx=np.nonzero(mask);tx,ty=oldx[mask],oldy[mask]
        values=np.zeros(mask.shape,np.float32);valid=np.zeros(mask.shape,bool)
        values[ty,tx]=d[mask];valid[ty,tx]=True
        reverse_prior=nearest_prior(values,valid)
    tx,ty,cost=periodic_direction(left,right,left_valid,right_valid,radius,1,forward_prior)
    bx,by,_=periodic_direction(right,left,right_valid,left_valid,radius,-1,reverse_prior)
    y,x=np.indices(left_valid.shape);cx=np.clip(tx,0,x.shape[1]-1);cy=np.clip(ty,0,y.shape[0]-1)
    valid=(tx>=0)&(ty>=0)&(np.abs(bx[cy,cx]-x)<=3)&(np.abs(by[cy,cx]-y)<=3)
    disparity=(x-tx).astype(np.float32);disparity[~valid]=np.nan
    return disparity,cost,np.abs(ty-y).astype(np.float32),ty,tx


def hybrid_matches(left,right,left_valid,right_valid,colors,vertical_radius=0):
    """Spatial regularization on two decoded projector axes + natural intensity."""
    coords=[projector_coordinates(d) for d in (left,right)]
    images=[]
    for uv,color in zip(coords,colors):
        im=np.dstack((uv,cv.cvtColor(color,cv.COLOR_BGR2GRAY)))
        images.append(np.rint(np.clip(im,0,255)).astype(np.uint8))
    span=768
    def compute(a,b,minimum):
        solver=cv.StereoSGBM_create(minDisparity=minimum,numDisparities=span,blockSize=3,
            P1=216,P2=864,uniquenessRatio=1,disp12MaxDiff=3,
            speckleWindowSize=40,speckleRange=3,mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
        return solver.compute(a,b).astype(np.float32)/16
    d=compute(images[0],images[1],0);rev=compute(images[1],images[0],1-span)
    y,x=np.indices(d.shape);xr=np.rint(x-d).astype(np.int32);safe=np.clip(xr,0,d.shape[1]-1)
    if vertical_radius:
        tx,ty,cost=local_2d_search(coords[0],coords[1],left_valid,right_valid,xr,d>0,vertical_radius)
        reverse_x=np.rint(x-rev).astype(np.int32)
        back_x,back_y,_=local_2d_search(coords[1],coords[0],right_valid,left_valid,
                                      reverse_x,(rev>-span)&(rev<0),vertical_radius)
        safe_x=np.clip(tx,0,d.shape[1]-1);safe_y=np.clip(ty,0,d.shape[0]-1)
        valid=(tx>=0)&(ty>=0)&(x-tx>1)&(x-tx<span)&(cost<=6)
        valid &= (np.abs(back_x[safe_y,safe_x]-x)<=3)&(np.abs(back_y[safe_y,safe_x]-y)<=3)
        disparity=(x-tx).astype(np.float32);disparity[~valid]=np.nan
        return disparity,cost,np.abs(ty-y).astype(np.float32),ty,tx
    diff=np.abs(coords[0]-coords[1][y,safe]);diff=np.minimum(diff,256-diff)
    cost=np.max(diff,axis=-1)
    valid=(d>0)&(xr>=0)&(xr<d.shape[1])&left_valid&right_valid[y,safe]
    valid &= (rev[y,safe]>-span)&(np.abs(d+rev[y,safe])<=3)&(cost<=12)
    d[~valid]=np.nan
    return d,cost,np.zeros(d.shape,np.float32),y.astype(np.int32),xr


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,default=root/'captures/session_20261017_153222_240563')
    parser.add_argument('--output',type=Path,default=root/'results/session_153222')
    parser.add_argument('--width',type=int,default=1920)
    parser.add_argument('--method',choices=['temporal','hybrid','periodic'],default='hybrid')
    parser.add_argument('--prior-dir',type=Path,default=root/'results/session_153222_vertical40',
                        help='Optional soft disparity preference; does not limit which pixels are searched')
    parser.add_argument('--vertical-radius',type=int,default=0,
                        help='Search +/- this many rows in rectified processing pixels')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if not 0<=args.vertical_radius<=100:
        parser.error('--vertical-radius must be between 0 and 100')
    if args.vertical_radius and args.method=='temporal':
        parser.error('--vertical-radius requires --method hybrid or periodic')
    calibration=root.parent/'lab4/results/self_written/calibration/calibration.json'
    saved=json.loads(calibration.read_text(encoding='utf-8'))
    original={key:np.array(saved[key],float) for key in ('K1','D1','K2','D2','R','T')}
    first=read(args.session/'l1.bmp');h,w=first.shape[:2]
    if [w,h]!=saved['image_size']:
        raise ValueError('Calibration image dimensions differ')
    size=(args.width,round(h*args.width/w));scale=args.width/w
    c={key:value.copy() for key,value in original.items()}
    for key in ('K1','K2'):
        c[key][:2]*=scale
    # Average complementary Gray patterns removes most projected texture.
    means=[]
    for role in ('l','r'):
        total=np.zeros((size[1],size[0],3),np.float32)
        for capture in KEPT[:10]:
            total+=cv.resize(read(args.session/f'{role}{capture}.bmp'),size,interpolation=cv.INTER_AREA)
        means.append(np.rint(total/10).astype(np.uint8))
    sift=cv.SIFT_create(nfeatures=12000)
    kp1,de1=sift.detectAndCompute(means[0],None);kp2,de2=sift.detectAndCompute(means[1],None)
    matches=[a for a,b in cv.BFMatcher().knnMatch(de1,de2,k=2) if a.distance<.7*b.distance]
    p=np.float32([kp1[m.queryIdx].pt for m in matches]);q=np.float32([kp2[m.trainIdx].pt for m in matches])
    _,fm=cv.findFundamentalMat(p,q,cv.USAC_MAGSAC,1.5,.999)
    p,q=p[fm.ravel()!=0],q[fm.ravel()!=0]
    u=cv.undistortPoints(p[:,None],c['K1'],c['D1']).reshape(-1,2)
    v=cv.undistortPoints(q[:,None],c['K2'],c['D2']).reshape(-1,2)
    E,em=cv.findEssentialMat(u,v,np.eye(3),method=cv.RANSAC,prob=.999,threshold=1./c['K1'][0,0])
    count,R,T,pose_mask=cv.recoverPose(E,u,v,np.eye(3),mask=em)
    if count<30:
        raise ValueError('Insufficient inliers to refine current scene pose')
    baseline=float(np.linalg.norm(c['T']))
    c['R']=R;c['T']=T*baseline
    geom=cv.stereoRectify(c['K1'],c['D1'],c['K2'],c['D2'],size,R,c['T'],
                         flags=cv.CALIB_ZERO_DISPARITY,alpha=0)
    a=cv.undistortPoints(p[pose_mask.ravel()!=0,None],c['K1'],c['D1'],R=geom[0],P=geom[2]).reshape(-1,2)
    b=cv.undistortPoints(q[pose_mask.ravel()!=0,None],c['K2'],c['D2'],R=geom[1],P=geom[3]).reshape(-1,2)
    residual=np.abs(a[:,1]-b[:,1])
    print('Scene pose inliers:',count,'vertical median/p90',np.percentile(residual,[50,90]),flush=True)
    stacks=[];colors=[]
    for side,role in enumerate(('l','r')):
        maps=cv.initUndistortRectifyMap(c[f'K{side+1}'],c[f'D{side+1}'],geom[side],geom[side+2],size,cv.CV_32FC1)
        images=[]
        for capture in KEPT:
            gray=cv.cvtColor(cv.resize(read(args.session/f'{role}{capture}.bmp'),size,interpolation=cv.INTER_AREA),cv.COLOR_BGR2GRAY)
            images.append(cv.remap(gray,*maps,cv.INTER_LINEAR).astype(np.float32))
        stacks.append(np.stack(images))
        colors.append(cv.remap(means[side],*maps,cv.INTER_LINEAR))
        picture(out/f'{role}_rectified_mean.png',colors[-1])
    desc1,mask1,mod1=signature(stacks[0]);desc2,mask2,mod2=signature(stacks[1])
    del stacks
    picture(out/'modulation_gray.png',np.clip(mod1*3,0,255).astype(np.uint8))
    if args.method=='periodic':
        d,cost,yerror,yr,xr=periodic_matches(desc1,desc2,mod1>=4,mod2>=4,
                                            args.vertical_radius,args.prior_dir)
    elif args.method=='hybrid':
        d,cost,yerror,yr,xr=hybrid_matches(desc1,desc2,mod1>=4,mod2>=4,colors,args.vertical_radius)
    else:
        d,cost,yerror,yr,xr=dense_matches(desc1,desc2,mask1,mask2,args.width*.45)
    del desc1,desc2
    valid=np.isfinite(d)
    # Reject disconnected tiny patches and local disparity spikes, no hole filling.
    median=cv.medianBlur(np.nan_to_num(d,nan=0),5)
    valid &= (np.abs(np.nan_to_num(d)-median)<=3)
    n,labels,stats,_=cv.connectedComponentsWithStats(valid.astype(np.uint8),8)
    keep=np.zeros(n,bool);keep[1:]=stats[1:,cv.CC_STAT_AREA]>=40
    valid &= keep[labels]
    yy,xx=np.nonzero(valid)
    # Use actual matched right y in triangulation instead of forcing equal rows.
    homogeneous=cv.triangulatePoints(geom[2],geom[3],np.float32([xx,yy]),np.float32([xr[yy,xx],yr[yy,xx]]))
    points=(homogeneous[:3]/homogeneous[3]).T
    accepted=np.isfinite(points).all(axis=1)&(points[:,2]>0)&(points[:,2]<500)
    valid[yy[~accepted],xx[~accepted]]=False
    yy,xx=yy[accepted],xx[accepted];points=points[accepted]
    depth=np.full(d.shape,np.nan,np.float32);depth[yy,xx]=points[:,2]
    rgb=colors[0][yy,xx,::-1]
    original_points=points@geom[0]
    ply(out/'colored_cloud.ply',original_points,rgb)
    # camera intrinsics are scaled to processing size, same world coordinates.
    cp,cr=cameras(c,size)
    ply(out/'final.ply',np.vstack((original_points,cp)),np.vstack((rgb,cr)))
    for name,array in [('depth',depth),('disparity',d),('valid_mask',valid),('temporal_cost',cost),
                        ('matched_right_y',yr),('matched_right_x',xr),('vertical_search_offset',yr-np.indices(d.shape)[0])]:
        np.save(out/f'{name}.npy',array)
    picture(out/'depth.png',gray_preview(depth,valid))
    picture(out/'valid_mask.png',valid.astype(np.uint8)*255)
    # Projected-object mask is an explicit image ROI, not an estimated segmentation.
    roi=np.zeros(valid.shape,np.uint8)
    # Map original left-image skull bounding rectangle into the rectified frame.
    corners=np.float32([[.36*w,.18*h],[.66*w,.18*h],[.66*w,.59*h],[.36*w,.59*h]])
    rect=cv.undistortPoints((corners*scale)[:,None],c['K1'],c['D1'],R=geom[0],P=geom[2]).reshape(-1,2)
    cv.fillConvexPoly(roi,np.rint(rect).astype(np.int32),1)
    focus=valid&(roi>0);selected=roi[yy,xx]>0
    ply(out/'skull_cloud.ply',original_points[selected],rgb[selected])
    picture(out/'skull_depth.png',gray_preview(depth,focus))
    from run_test import point_preview
    point_preview(out/'skull_cloud.ply',out/'skull_point_cloud_preview.jpg')
    np.savez(out/'geometry_used.npz',**c,R1=geom[0],R2=geom[1],P1=geom[2],P2=geom[3],Q=geom[4])
    metrics=dict(processing_size=size,points=len(points),skull_roi_points=int(selected.sum()),
        valid_fraction=float(valid.mean()),pose_inliers=count,
        scene_vertical_median_px=float(np.median(residual)),scene_vertical_p90_px=float(np.percentile(residual,90)),
        scale='Inherited baseline in checker-square units; baseline unchanged is an assumption, not independently verified',
        baseline=baseline,excluded_duplicate_captures=[2,11],capture_order=KEPT,
        interpolation=False,gray_depth='near bright, far dark, invalid black; 2%-98% display range',
        vertical_search_radius_px=args.vertical_radius,
        matching_details=(dict(phase_representation='cos/sin of each 128px phase period',
            gray_bit_weight=.18,nearest_candidates=160,max_phase_difference_px=4,
            max_gray_bit_mismatches=2,bidirectional_return_tolerance_px=3,
            initial_seed_required=False,prior_usage='soft ranking only; no inferred point exported')
            if args.method=='periodic' else None),
        vertical_offset_p10_p50_p90=np.percentile((yr-np.indices(d.shape)[0])[valid],[10,50,90]).tolist(),
        method=args.method+'; both-axis Gray-bit and four-step phase signatures, epipolar correspondence, triangulation',
        limitations='Only Gray bits 6..2 captured. Codes repeat; matching relies on scene pose and is not a full absolute projector calibration.')
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'REPORT.md').write_text(
        '# 本次结构光重建\n\n'
        '重复拍摄编号2、11已移到原会话的 excluded_duplicates；其余28对按PPT顺序使用。\n\n'
        '从两个方向的5位格雷码差分及四步相移提取时序描述子，按当前场景极线约束寻找对应，'
        '过滤不可靠小块后逐有效像素三角化，没有通过插值增加点数。\n\n'
        f'处理尺寸 {size}，场景点 {len(points)}，头骨图像ROI内 {selected.sum()}。\n\n'
        '![灰度深度](depth.png)\n\n![头骨灰度深度](skull_depth.png)\n\n'
        '![头骨点云](skull_point_cloud_preview.jpg)\n\n'
        '- final.ply：场景及双相机。\n- colored_cloud.ply：场景彩色点云。\n'
        '- skull_cloud.ply：左图头骨包围框对应的点云，边缘可能含背景，并非精确物体分割。\n'
        '- depth.npy：原始深度，无效为NaN。depth.png：灰度显示，近亮远暗。\n\n'
        '## 几何与尺度限制\n\n'
        '旧标定与本次画面存在明显极线残差，故保留旧内参，并从本次图案互补均值图估计相机R/T方向。'
        f'场景姿态内点 {count}，校正后竖直残差中位数 {np.median(residual):.3f}px。'
        '基线长度沿用旧标定，只有基线未改变时尺度才成立；单位是棋盘格边长，不能称为毫米精度测量。'
        '建议重新采集同一固定装置的双目标定以验证三维形状。\n\n'
        '当前删去了格雷码高位，仅保留bit6到bit2，编码存在周期歧义，横竖联合和几何约束能够减少歧义，'
        '但不能保证全部匹配正确。阴影、反光与低调制度区域仍有空洞。\n',encoding='utf-8')
    if args.method=='periodic':
        with (out/'REPORT.md').open('a',encoding='utf-8') as stream:
            stream.write('\n## 周期边界修正\n\n'
                '以两方向相位的cos/sin表示进行候选搜索，最终以atan2(cross,dot)计算圆周相位差；'
                '不再用解码后的0–255坐标生成SGBM初始候选。格雷码位差低权参与排序，允许最多2位不同。'
                '全部调制度合格像素均可搜索，不要求已有视差；旧视差仅作弱排序参考，不作深度填充。'
                f'纵向范围±{args.vertical_radius}像素，水平视差在1–768之间，'
                '候选相位差≤4个投影周期像素，双方向返回距离横纵均≤3像素。'
                '保持原有局部视差和小连通域过滤；未使用深度或点云插值。\n')
    print(json.dumps(metrics,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
