import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from matplotlib.backends.backend_agg import FigureCanvasAgg


def fig_to_numpy(fig: plt.Figure, normalize=True):
    fig.tight_layout(pad=0)
    # Draw figure on canvas
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    img = np.frombuffer(s, np.uint8).reshape((height, width, 4))[:, :, :3]

    # Convert the figure to numpy array, read the pixel values and reshape the array
    # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = np.swapaxes(img, 0, 2)  # CHW
    if normalize:
        img = img / 255.0
    return img


def text_on_img(img, text, fill=(255, 255, 255)):
    assert isinstance(img, torch.Tensor)
    img_with_text = Image.fromarray(img.transpose(0, 2).transpose(0, 1).cpu().numpy(), mode='RGB')
    d = ImageDraw.Draw(img_with_text)
    d.text((2, 2), text, fill=fill, )
    img_with_text = torch.tensor(np.array(img_with_text), device=img.device).transpose(0, 2).transpose(1, 2)
    return img_with_text


def text_on_imgs(imgs, texts, fill=(255, 255, 255)):
    imgs_with_texts = []
    for img, text in zip(imgs, texts):
        imgs_with_texts.append(text_on_img(img, text, fill))
    imgs_with_texts = torch.stack(imgs_with_texts)
    return imgs_with_texts


def pallate_img_tensor_uint8(gray_img, pallate=None):
    # in and out range is 0~255
    assert gray_img.ndim == 4 and gray_img.shape[1] == 1  # channel of gray image should be 1
    if pallate is None:
        pallate = PALLATE
    pallate = torch.tensor(pallate, dtype=torch.float32, device=gray_img.device).view(-1, 3)
    if gray_img.min() < 0. or gray_img.max() > 255.:
        raise RuntimeError('input gray img should be in range 0~255')
    gray_img = gray_img.type(torch.long)
    colored_img = pallate[gray_img].transpose(1, -1).squeeze(-1)
    return colored_img


def pallate_img_tensor(gray_img: torch.Tensor, pallate=None, input_range=(-1, 1), output_range=(-1, 1)):
    assert gray_img.ndim == 3  # BHW
    gray_img = gray_img.unsqueeze(1)
    if pallate is None:
        pallate = PALLATE
    pallate = torch.tensor(pallate, dtype=torch.float32, device=gray_img.device).view(-1, 3)
    gray_img = gray_img.to(torch.float32)
    gray_img = gray_img.clip(input_range[0], input_range[1])
    gray_img = 255 * (gray_img - input_range[0]) / (input_range[1] - input_range[0])
    gray_img = gray_img.to(torch.long)
    colored_img = pallate[gray_img].transpose(1, -1).squeeze(-1)
    colored_img = colored_img.to(torch.float32) / 255
    colored_img = colored_img * (output_range[1] - output_range[0]) + output_range[0]
    colored_img = colored_img.to(torch.float32)
    return colored_img


def gray2rgb(gray_img, dtype=torch.float32):
    assert gray_img.ndim == 4
    if gray_img.shape[1] == 1:
        rgb_img = gray_img.repeat([1, 3, 1, 1])
    elif gray_img.shape[1] == 3:
        rgb_img = gray_img
    else:
        raise AttributeError
    if dtype == torch.uint8:  # input -1~1
        rgb_img = torch.clip(rgb_img, -1, 1)
        rgb_img = (127.5 * (rgb_img + 1)).type(torch.uint8)
    elif dtype != rgb_img.dtype:
        rgb_img = rgb_img.type(dtype)
    return rgb_img


PALLATE = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0,
           192, 0,
           0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0,
           0, 192, 0,
           128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0, 192, 64, 0, 64, 192, 0,
           192, 192, 0,
           64, 64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128, 64, 128, 128, 64,
           0, 0, 192,
           128, 0, 192, 0, 128, 192, 128, 128, 192, 64, 0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192,
           192, 0, 192,
           64, 128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0, 64, 192, 128, 64, 192,
           0, 192,
           192, 128, 192, 192, 64, 64, 64, 192, 64, 64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64,
           192, 192,
           192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0, 128, 32, 128, 128, 160,
           128, 128,
           96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32,
           64, 0, 160,
           64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192, 128, 160, 192, 128, 96, 64, 0, 224,
           64, 0, 96,
           192, 0, 224, 192, 0, 96, 64, 128, 224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32,
           128, 64,
           160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128, 192, 96, 0, 64, 224, 0, 64, 96, 128, 64,
           224, 128,
           64, 96, 0, 192, 224, 0, 192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192,
           64, 32,
           64, 192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64,
           96, 64,
           192, 224, 64, 192, 96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32, 128,
           128, 32,
           128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32,
           128, 64,
           160, 128, 192, 160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0, 224,
           128, 128,
           224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192,
           224, 128,
           0, 32, 64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128, 160, 192,
           64, 32, 64,
           192, 32, 64, 64, 160, 64, 192, 160, 64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96,
           64, 128, 96,
           64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192, 128, 224, 192, 64, 96, 64, 192, 96,
           64, 64,
           224, 64, 192, 224, 64, 64, 96, 192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32,
           160, 0,
           160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160, 128, 96, 32, 0, 224, 32, 0, 96, 160, 0,
           224, 160,
           0, 96, 32, 128, 224, 32, 128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160, 224,
           0, 32, 96,
           128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0, 224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96,
           128, 224,
           96, 128, 96, 224, 128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192,
           160, 32,
           192, 32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224,
           32, 192,
           96, 160, 192, 224, 160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96,
           192, 32,
           224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192,
           96, 224,
           192, 224, 224, 192]
