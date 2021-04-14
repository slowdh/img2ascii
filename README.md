# img2ascii

## How it works?
 <img src="https://github.com/slowdh/img2ascii/blob/main/test_image.jpg">

It takes video input and outputs ascii converted video.
Converting step is done with resizing image, and mapping each pixel to specific character according to brightness. (now 10 steps.)
If you want to change character combinations, check convert() function.



## Examples

```python
from img2ascii import capture_and_convert

capture_and_convert(size=(640, 480), resize_factor=1, source_path='test_input.mp4', save_path='test_output.mp4')

```
