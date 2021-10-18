img_sz = 5000
window_size = 512

def get_mx(stride: int) -> int:
    mx_x1 = 0

    for x0 in range(0, img_sz, stride):
        x1 = x0 + window_size
        if x1 <= 5000:
            mx_x1 = x1

    return mx_x1

stride = window_size - 1
while get_mx(stride) != 5000:
    stride -= 1

print(stride)