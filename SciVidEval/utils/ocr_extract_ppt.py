import os.path

from cnocr import CnOcr


def frame2txt(img_path, state=None):
    # img_path = img_path
    img_path = os.path.join(state['frame_dir'], img_path)
    # print(img_path)
    model = CnOcr()
    # print(model)
    result = model.ocr(img_path)
    output = ''
    for item in result:
        text = item['text']
        if '西安电子科技大学' in text: continue
        output += text + '\n'
    return output


# if __name__ == '__main__':
#     result = frame2txt('./temp/frame/frame_0001.jpg')
#     print(result)