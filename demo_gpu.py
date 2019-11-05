# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from tools.test import *

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

###Modify###

###Modify###

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('using cuda')
    else:
        print('using cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    ###Modify###
    VeryBig=999999999  # 用于将视频框调整到最大
    Cap = cv2.VideoCapture(0)
    ret, frame = Cap.read()  # 读取帧
    
    #ims = [frame] # 把frame放入列表格式的frame， 因为原文是将每帧图片放入列表
    ###Modify###

    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    ###Modify###展示当前图片，当按下space ，进入框选阶段
    getROI = 0
    while(getROI == 0):
        ret, frame = Cap.read()
        cv2.imshow('frame',frame)
        #key = cv2.waitKey(5)
        #print('%d' % key)
        if cv2.waitKey(100) == 32: #space
            print('ready to select')
            init_rect = cv2.selectROI('SiamMask', frame, False, False)
            x, y, w, h = init_rect
            getROI = 1
    ###Modify###
    
    #try:
     #   init_rect = cv2.selectROI('SiamMask', frame, False, False)
        #init_rect = cv2.selectROI('SiamMask', frame, False, False)          
      #  x, y, w, h = init_rect
    #except:
     #       exit()

    toc = 0

    ###Modify###
    im=frame
    f=0
    target_pos = np.array([x + w / 2, y + h / 2])
    target_sz = np.array([w, h])
    #img =torch.tensor(im, device=device).float()
    state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'],device=device)  # init tracker
    while(True):
        tic = cv2.getTickCount()
        ret, im = Cap.read()  # 逐个提取frame
        if (ret==False):
            break;
        #img =torch.tensor(im, device=device).float()
        state = siamese_track(state, im, mask_enable=True, refine_enable=True,device=device)  # track
        location = state['ploygon'].flatten()
        mask = state['mask'] > state['p'].seg_thr
 
        im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
        cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
        cv2.imshow('SiamMask', im)
        key = cv2.waitKey(1)
        if key > 0:
            break
 
        toc += cv2.getTickCount() - tic
        f=f+1
    ###Modify###

    ###Modify###下面被替换
    #for f, im in enumerate(ims):
     #   tic = cv2.getTickCount()
      #  if f == 0:  # init
       #     target_pos = np.array([x + w / 2, y + h / 2])
        #    target_sz = np.array([w, h])
         #   state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        #elif f > 0:  # tracking
         #   state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
         #  location = state['ploygon'].flatten()
     #       mask = state['mask'] > state['p'].seg_thr

     #      im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
     #      cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
      #      cv2.imshow('SiamMask', im)
       #     key = cv2.waitKey(1)
        #    if key > 0:
         #       break

        #toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
if __name__ == '__main__':
    main()
