from resize_image import resize_image
from convert_out_to_standard import convert_out_to_standard
from quick_draw import predict_image

def combine_quickdraw():
    resize_image()
    convert_out_to_standard()
    result = predict_image()
    return result

# print(combine_quickdraw())