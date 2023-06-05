import random

import cv2
import numpy as np
import torch
import os
import torch.nn as nn

from utils.utils import *
from functools import lru_cache

_VIDEO_EXT = ['.avi', '.mp4', '.mov']
_IMAGE_EXT = ['.jpg', '.png']
_IMAGE_SIZE = 224

## FlowUnderAttack load model
import sys
sys.path.append(f'./FlowUnderAttack/')
sys.path.append(f'./FlowUnderAttack/flow_library')
import helper_functions.ownutilities as ownutilities
from helper_functions.ownutilities import show_images
net = 'RAFT'#'FlowNetC'#
custom_weight_path = './FlowUnderAttack/models/_pretrained_weights/raft-sintel.pth'#FlowNet2-C_checkpoint.pth.tar'#

model_takes_unit_input = ownutilities.model_takes_unit_input(net)
model, path_weights = ownutilities.import_and_load(net, custom_weight_path=custom_weight_path, make_unit_input=not model_takes_unit_input, variable_change=False, make_scaled_input_model=True,device='cuda')
model.eval()
for p in model.parameters():
    p.requires_grad = False


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(n))]


def resize_img(img, short_size=256):
    h, w, c = img.shape
    if (w <= h and w == short_size) or (h <= w and h == short_size):
        return img
    if w < h:
        ow = short_size
        oh = int(short_size * h / w)
        return cv2.resize(img, (ow, oh))
    else:
        oh = short_size
        ow = int(short_size * w / h)
        return cv2.resize(img, (ow, oh))


def video_loader(video_path, short_size):
    log("Start processing video:", video_path)
    video = []
    vidcap = cv2.VideoCapture(str(video_path))
    major_ver, *_ = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = vidcap.get(cv2.CAP_PROP_FPS)
    success, image = vidcap.read()
    video.append(resize_img(image, short_size))
    while success:
        success, image = vidcap.read()
        if not success: break
        video.append(resize_img(image, short_size))
    vidcap.release()

    return video, len(video), fps


def images_loader(images_path, transform=None):
    log("Start processing Image set:", images_path)
    images_set = []
    images_list = [i for i in images_path.iterdir() if not i.stem.startswith('.') and i.suffix.lower() in _IMAGE_EXT]
    images_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f.stem))))
    for image_path in images_list:
        image = cv2.imread(str(image_path), 3)
        if transform is not None:
            image = transform(image)
        images_set.append(image)
    return images_set, len(images_set)


def sample_by_number(frame_num, out_frame_num, random_choice=False):
    full_frame_lists = split(list(range(frame_num - 1)), out_frame_num)
    if random_choice:
        return [random.choice(i) for i in full_frame_lists]
    else:
        return [i[0] for i in full_frame_lists]


def sample_by_fps(frame_num, in_fps, out_fps, random_choice=False):
    if in_fps is not None:
        out_frame_num = int(frame_num * out_fps / in_fps)
    full_frame_lists = split(list(range(frame_num - 1)), out_frame_num)
    if random_choice:
        return [random.choice(i) for i in full_frame_lists]
    else:
        return [i[0] for i in full_frame_lists]


class FrameGenerator(object):
    def __init__(self, input_path,
                 sample_num=1,
                 random_choice=False,
                 use_fps=True,
                 resize=None,
                 in_fps=30,
                 out_fps=5):
        """
        :param input_path: The input video file or image set path
        :param sample_num: The number of frames you hope to use, they are chosen evenly spaced
        :param slice_num: The number of blocks you want to divide the input file into, and frames
                            are randomly chosen from each block.
        """
        input_path = Path(input_path)
        self.is_video = input_path.is_file() and input_path.suffix.lower() in _VIDEO_EXT
        if self.is_video:
            self.frames, self.frame_num, self.fps = video_loader(input_path, resize)
        elif input_path.is_dir():
            self.frames, self.frame_num = images_loader(input_path, resize)
        else:
            raise IOError("Input data path is not valid! Please make sure it is whether "
                          "a video file or a image set directory")
        self.counter = 0
        self.current_video_frame = -1
        self.sample_num = sample_num

        if use_fps:
            self.chosen_frames = sample_by_fps(self.frame_num, in_fps, out_fps, random_choice)
        else:
            self.chosen_frames = sample_by_number(self.frame_num, sample_num + 1, random_choice)

    def __len__(self):
        return len(self.chosen_frames)

    def reset(self):
        self.counter = 0

    def get_frame(self):
        frame = self.frames[self.counter]  # cv2.resize(frame, (_IMAGE_SIZE, _IMAGE_SIZE))
        self.counter += 1
        return frame


def get_video_generator(video_path, opts, class_name=None):
    if opts.out_path is None:
        out_path = Path(*video_path.parts[:-3], "pre-processed", "train",
                        video_path.parts[-2], video_path.stem)
        if not out_path.exists(): out_path.mkdir()
    else:
        out_path = Path(opts.out_path)
    out_path_dic = {}

    out_path_dic["flow"] = out_path/"flow"/class_name if class_name is not None else out_path/"flow"
    os.makedirs(out_path_dic["flow"], exist_ok=True)
    out_path_dic["rgb"] = out_path/"rgb"/class_name if class_name is not None else out_path/"rgb"
    os.makedirs(out_path_dic["rgb"], exist_ok=True)

    video_object = FrameGenerator(video_path, opts.sample_num, opts.random_choice, use_fps=(opts.sample_type == "fps"),
                                  resize=opts.resize, in_fps=opts.in_fps, out_fps=opts.out_fps)
    return video_object, out_path_dic


def compute_rgb(video_object, out_path):
    """Compute RGB"""
    # for i in range(len(video_object) - 1):
    #     frame = video_object.get_frame()
    #     # Kind of like the normalization
    #     frame = (frame / 255.)# * 2 - 1
    #     rgb.append(frame)
    # # rgb = rgb[:-1]
    # rgb = np.float32(np.array(rgb))
    # rgb = np.array(video_object.frames[:-1])
    # np.save(out_path["rgb"], rgb)
    # log('save rgb with shape ', rgb.shape)
    rgb = np.array([video_object.get_frame() for _ in range(len(video_object) - 1)])
    return rgb

@lru_cache(maxsize=1)
def get_patch_and_defense(
    patch = '',
    defense = '',
    ):
    if patch != '':
        from helper_functions.patch_adversary import PatchAdversary
        P = PatchAdversary(patch, size = 100, angle = [-10,10], scale = [.95,1.05]) # Change of variable is always false for evaluation
        P = P.cuda().requires_grad_(False)
    else:
        P = lambda x,y: (x,y,1)

    if defense != '' or defense.lower() != 'none':
        from helper_functions.defenses import ILP,LGS
        if defense == 'ILP':
            print(f"ILP defense with k = {16}, o = {8}, t = {.15}, s = {15}, r = {5}")
            D = ILP(16, 8, .15, 15, 5, "forward")
        elif defense == 'LGS':
            print(f"LGS defense with k = {16}, o = {8}, t = {.15}, s = {15}")
            D = LGS(16, 8, .15, 15, "forward")
        else:
            print("No defense")
            D = lambda x,y: (x,y)
    return P,D

def compute_flow(video_object, opts, upsample_factor=2.3):
    """Compute the TV-L1 optical flow."""

    P,D = get_patch_and_defense(opts.patch_path, opts.defense)
    flow = []

    bins = np.linspace(-20, 20, num=256)
    frame1 = video_object.get_frame()
    frame1 = torch.from_numpy(frame1).permute(2, 0, 1).unsqueeze(0).float().cuda().requires_grad_(False)
    for i in range(len(video_object) - 1):
        frame2 = video_object.get_frame()
        frame2 = torch.from_numpy(frame2).permute(2, 0, 1).unsqueeze(0).float().cuda().requires_grad_(False)


        shape = frame1.shape
        I1 = nn.Upsample(scale_factor=upsample_factor, mode='bilinear')(frame1)
        I2 = nn.Upsample(scale_factor=upsample_factor, mode='bilinear')(frame2)
        padder, [I1, I2] = ownutilities.preprocess_img(net, I1, I2)
        if not model_takes_unit_input:
            I1 = I1 / 255.
            I2 = I2 / 255.
            
        I1, I2,*_ = P(I1,I2)
        I1, I2 = D(I1,I2)
            
        curr_flow = ownutilities.compute_flow(model,"scaled_input_model",I1,I2)
        [curr_flow] = ownutilities.postprocess_flow(net, padder, curr_flow)
        # downsample flow
        curr_flow = upsample_flow(curr_flow.permute(0,2,3,1).cpu(), shape[2], shape[3])
        curr_flow = curr_flow.numpy()[0]

        # Append this flow frame
        flow.append(curr_flow)

        frame1 = frame2

    flow = np.array(flow)  # np.float32(
    # np.save(out_path["flow"], flow)
    # log("Save flow with shape ", flow.shape)
    return flow


def compute_tvl1_flow(video_object, out_path):
    """Compute the TV-L1 optical flow."""
    flow = []

    bins = np.linspace(-20, 20, num=256)
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    frame1 = video_object.get_frame()
    prev = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    for i in range(len(video_object) - 1):
        frame2 = video_object.get_frame()
        curr = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        curr_flow = TVL1.calc(prev, curr, None)
        assert (curr_flow.dtype == np.float32)

        # Truncate large motions
        curr_flow[curr_flow >= 20] = 20
        curr_flow[curr_flow <= -20] = -20

        # curr_flow = np.digitize(curr_flow, bins)
        # curr_flow = (curr_flow / 255.) * 2 - 1

        # Append this flow frame
        flow.append(curr_flow)

        prev = curr

    flow = np.array(flow)  # np.float32(
    # np.save(out_path["flow"], flow)
    log("Save flow with shape ", flow.shape)
    return flow

def pre_process(video_path, opts, class_name=None):
    video_path = Path(video_path)
    
    with Timer('Loading video'):
        log('Loading video...')
        video_object, out_path_dic = get_video_generator(video_path, opts, class_name)

    with Timer('Compute RGB'):
        log('Extract RGB...')
        rgb_data = compute_rgb(video_object, out_path_dic)
        np.save(out_path_dic["rgb"]/f"{video_path.stem}.npy", rgb_data)

    video_object.reset()
    with Timer('Compute flow'):
        log('Extract Flow...')
        # flow_data = compute_tvl1_flow(video_object, opts)
        flow_data = compute_flow(video_object, opts)
        np.save(out_path_dic["flow"]/f"{video_path.stem}.npy", flow_data)
    return rgb_data, flow_data


def mass_process(opts):
    if opts.input_path is None:
        if opts.is_image:
            data_root = Path("data/images/")
        else:
            data_root = Path("data/videos/")
    else:
        data_root = Path(opts.input_path)
    class_paths = [i for i in data_root.iterdir() if not i.stem.startswith(".") and i.is_dir()]
    item_paths = []
    for class_path in class_paths:
        if opts.is_image:
            item_paths.extend([i for i in class_path.iterdir()
                               if not i.stem.startswith(".") and i.is_dir()])
        else:
            item_paths.extend([i for i in class_path.iterdir()
                               if not i.stem.startswith(".") and i.is_file() and i.suffix.lower() in _VIDEO_EXT])
    
    # create classes.txt
    os.makedirs(Path(opts.out_path), exist_ok=True)
    with open(Path(opts.out_path)/"classes.txt", "w") as f:
        for i in class_paths:
            f.write(f"{i.name}\n")

    for item_path in item_paths:
        class_name = item_path.parent.name
        # if already processed, skip
        
        if (Path(opts.out_path)/"rgb"/class_name/item_path.stem).exists() and \
                (Path(opts.out_path)/"flow"/class_name/item_path.stem).exists():
            log("Already processed:", str(item_path))
            continue
        
        with Timer(item_path.name):
            log("Now start processing:", str(item_path))
            pre_process(item_path, opts, class_name)


def main(opts):
    pre_process(opts.input_path, opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pre-process the video into formats which i3d uses.')

    # Main action arguments
    parser.add_argument(
        '--is_image',
        action='store_true',
        help='Use a series of image(its folder) as video input')
    parser.add_argument(
        '--mass',
        action='store_true',
        help='Compute RGBs and Flows massively.')
    parser.add_argument(
        '--init_dir',
        action='store_true',
        help='Initialize the data pre-processed folder tree.')
    parser.add_argument(
        '--input_path',
        type=str,
        default='data/videos/raw/take-out/IMG_6801.mp4',
        help='Path to input video or images folder')
    parser.add_argument(
        '--out_path',
        type=str,
        help='Where you want to save the output rgb and flow files.')

    # Sample arguments
    parser.add_argument(
        '--sample_num',
        type=int,
        default='16',
        help='The number of the output frames after the sample, or 1/sample_rate frames will be chosen.')
    parser.add_argument(
        '--in_fps',
        type=int,
        default='30',
        help="The FPS of the input image set. Video's FPS can be directly acquired.")
    parser.add_argument(
        '--out_fps',
        type=int,
        default='5',
        help='The FPS of the output video.')
    parser.add_argument(
        '--resize',
        type=int,
        default='256',
        help="Resize the short edge video to '--resize'. Mention that this is only the pre-process, random crop"
             "will be applied later when training or testing, so here 'resize' can be a little bigger.")
    parser.add_argument(
        '--random_choice',
        action='store_true',
        help='Whether to choose frames randomly or uniformly')
    parser.add_argument(
        '--sample_type',
        type=str,
        default='fps',
        help="'fps': sample the video to a certain FPS, or 'num': control the number of output video, "
             "choose the video sample method.")
    parser.add_argument(
        '--patch_path',
        type=str,
        default='',
        help="The path to the patch file, which is a png or npy file. If not specified, no patch will be applied.")
    parser.add_argument(
        '--defense',
        default='',
        choices=['', 'LGS','ILP'],
        help="The defense method to be applied. If not specified, no defense will be applied.")
    args = parser.parse_args()

    if args.is_image:
        DATA_ROOT = Path('data/images/')
    else:
        DATA_ROOT = Path('data/videos/')

    DATA_DIR = DATA_ROOT / 'raw'
    SAVE_DIR = DATA_ROOT / 'pre-processed'

    if args.init_dir:
        build_data_path(args.is_image)
        exit(0)
    if args.mass:
        mass_process(args)
    else:
        main(args)
