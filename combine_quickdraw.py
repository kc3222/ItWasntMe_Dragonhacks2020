from resize_image import resize_image
from convert_out_to_standard import convert_out_to_standard
from quick_draw import predict_image

def combine_quickdraw(path):
    resize_image(path, "new_out.csv")
    convert_out_to_standard("new_out.csv", "out_standard.csv")
    result = predict_image("out_standard.csv")
    return result

# print(combine_quickdraw())