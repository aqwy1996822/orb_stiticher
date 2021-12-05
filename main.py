import cv2
import numpy as np
import pytesseract
from PIL import Image

class Orb_stitch:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.frame_list = []

    def add_frame(self, frame):
        self.frame_list.insert(0, frame)

    def run(self):
        if len(self.frame_list) < 2:
            print("frame少于2")
            return -1

        res = self.frame_list[0]
        for index in range(1, len(self.frame_list)):
            frame = self.frame_list[index]
            kp1, kp2, good_match, match_image = self.match_res_frame(frame, res)
            res = self.stitch(kp1, kp2, good_match, res, frame)

            cv2.imshow("res_pad", res)
            # cv2.imshow("match_image", match_image)

            cv2.waitKey(1)
        print(res.shape)
        res=res[:,int(self.frame_list[0].shape[1]*0.6):res.shape[1]-int(self.frame_list[0].shape[1]*0.6),:]
        cv2.imwrite("res.png", res)
        text = self.ocr()
        res = cv2.putText(res, str(text),(10,50), cv2.FONT_ITALIC, 1, (0,0,255),2)
        cv2.imshow("res_pad", res)
        cv2.waitKey()
    def match_res_frame(self, frame, res):
        kp1 = self.orb.detect(res)
        kp2 = self.orb.detect(frame)

        # 计算描述符
        kp1, des1 = self.orb.compute(res, kp1)
        kp2, des2 = self.orb.compute(frame, kp2)

        # 对描述子进行匹配
        matches = self.bf.match(des1, des2)

        # 计算最大距离和最小距离
        min_distance = matches[0].distance
        for x in matches:
            if x.distance < min_distance:
                min_distance = x.distance

        # 筛选匹配点
        '''
            当描述子之间的距离大于两倍的最小距离时，认为匹配有误。
            但有时候最小距离会非常小，所以设置一个经验值30作为下限。
        '''
        good_match = []
        for x in matches:
            if x.distance <= max(2 * min_distance, 40):
                good_match.append(x)

        match_image = cv2.drawMatches(res, kp1, frame, kp2, good_match, outImg=None)

        return kp1, kp2, good_match, match_image

    def stitch(self, kp1, kp2, good_match, res, frame):
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)

        # 基于最近邻和随机取样一致性得到一个单应性矩阵
        M, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 0.4)

        if abs(M[0][0]-1)>0.2 or abs(M[1][1]-1)>0.2:
            return res
        result = cv2.warpPerspective(res, M, (frame.shape[1] + res.shape[1], frame.shape[0]+ res.shape[0]))

        result[:frame.shape[0], :frame.shape[1]] = frame

        result = self.remove_pad(result)
        return result
    def ocr(self):


        image = Image.open("res.png")
        code = pytesseract.image_to_string(image)
        code = code.replace("\n","")
        return code
    def remove_pad(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, bina = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)  # 调整裁剪效果
        indexes = np.where(bina == 255)  # 提取白色像素点的坐标
        left = min(indexes[0])  # 左边界
        right = max(indexes[0])  # 右边界
        width = right - left  # 宽度
        bottom = min(indexes[1])  # 底部
        top = max(indexes[1])  # 顶部
        height = top - bottom  # 高度

        return image[left:right, bottom:top, :]  # 图片截取


if __name__ == "__main__":
    orb_stitch = Orb_stitch()

    test = False
    if test:
        img1 = cv2.imread("video/1.png")
        img2 = cv2.imread("video/2.png")

        orb_stitch.add_frame(img1)
        orb_stitch.add_frame(img2)
        orb_stitch.run()
    cap = cv2.VideoCapture("video/test4.mp4")
    pad_h = 150
    pad_w = 20
    step = 10
    resize_rate = 0.5
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()

        h, w, _ = frame.shape
        frame = frame[pad_h:h - pad_h, pad_w:w - pad_w, :]
        reframe = cv2.resize(frame, (0, 0), fx=resize_rate, fy=resize_rate)
        cv2.imshow("res_pad", reframe)
        cv2.waitKey(1)
        if i % step != 0:
            continue

        orb_stitch.add_frame(reframe)

    orb_stitch.run()