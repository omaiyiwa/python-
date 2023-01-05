# ui
import numpy as np
from tkinter import *
import tkinter
import os
from time import time
import pandas as pd
from tkinter import ttk, messagebox
from tkinter.ttk import Style
import time
import configparser
from tkinter import filedialog
import warnings
from distutils.util import strtobool
import pyttsx3
import threading
import logging

#  detect
import cv2
import argparse
import torch
import random
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, scale_coords, increment_path
from utils.datasets import LoadImages
from pathlib import Path

warnings.filterwarnings("ignore")
selected_file_path = None
selected_folder = None
path = 'data\\images\\'
global device, model, imgsz, half, opt, colors, source, stride, names, save_txt, root


# ---------------------------------------给按钮添加注释-----------------------------------------
class ToolTip(object):
    """
    给按钮添加说明
    """

    def __init__(self, widget):
        self.widget = widget
        self.tip_window = None
        self.id = None
        self.x = self.y = 0
        self.text1 = None

    def showtip(self, text):
        self.text1 = text
        if self.tip_window or not self.text1:
            return
        x1, y1, cx, cy = self.widget.bbox("insert")
        x1 = x1 + self.widget.winfo_rootx() + 5
        y1 = y1 + cy + self.widget.winfo_rooty() - 17  # 提示框所在位置
        self.tip_window = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x1, y1))
        label1 = Label(tw, text=self.text1, justify=LEFT,
                       background="#ffffe0", relief=SOLID, borderwidth=1,
                       font=("楷体", "10"))
        label1.pack(ipadx=1)

    def hidetip(self):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()


def CreateToolTip(widget, text):
    """
    绑定光标进入离开
    """
    tool_tip = ToolTip(widget)

    def enter(event):
        tool_tip.showtip(text)

    def leave(event):
        tool_tip.hidetip()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


# ---------------------------------------yolov5 5.0版本画图-----------------------------------------
def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# ---------------------------------------企业官网-----------------------------------------
def sm_web():
    url = 'http://www.shanma.tech/'
    os.system('start ' + url)


class Application(Frame):
    def __init__(self, master=None):
        super().__init__(master)  # super()代表的是父类的定义 ，而不是父类的对像
        # self.master = master
        # self.pack()
        self.table = None
        self.SendText = None
        self.columns = None
        self.progressbar = None
        self.select_path = StringVar()
        self.select_model_path = StringVar()
        self.create_window()

    # ---------------------------------------删除某行-----------------------------------------
    def del_some(self):
        iid1 = self.table.selection()
        if iid1 == ():
            self.SendText.delete('1.0', END)
            self.SendText.insert(END, '请先选择要删除的数据!\n\n')
            self.SendText.see(END)
            self.SendText.update()
        else:
            for it in iid1:
                item_text = self.table.item(it, "values")
                self.SendText.delete('1.0', END)
                self.SendText.insert(END, '已删除了{}的数据!\n\n'.format(item_text[0]))
                self.SendText.see(END)
                self.SendText.update()
            self.table.delete(iid1)

    def del_some_cancel(self):
        t2 = threading.Thread(target=self.del_some, args=())
        t2.setDaemon(True)
        t2.start()

    # ---------------------------------------保存数据-----------------------------------------
    def align_center(self, x):
        """
        保存excel自动居中
        """
        return ['text-align: center' for x in x]

    def save_result(self):
        """
        将获取到的光谱water, fat数值进行存储
        :return: 存储着结果的csv文件
        """
        global path
        os.makedirs('result', exist_ok=True)
        lst = []
        for row_id in self.table.get_children():
            row = self.table.item(row_id, 'values')
            jj = ','.join(row)  # 文本格式转数字格式
            ss = jj.split(',')
            for jjj in range(1, len(ss)):
                ss[jjj] = float(ss[jjj])
            lst.append(ss)
        try:
            a_pd = pd.DataFrame(lst)
            timesimply = time.strftime("_%H_%M_%S", time.localtime(time.time()))
            # 采用with建立空段, 出错才能os.remove
            with pd.ExcelWriter('./result/{}'.format(str(path.split("\\")[-3: -1][1]))
                                + str(timesimply) + '.xlsx') as writer:  # write an excel file
                # 自动居中对齐
                a_pd.style.apply(self.align_center, axis=0).to_excel(writer, 'sheet1',
                                                                     index=False,
                                                                     header=self.columns)  # write in ro file
                writer1 = writer.sheets['sheet1']  # 必须先定义保存excel的sheet_name
                # writer1.set_column("A:A", 10)  # 设置A列的宽度
                # writer1.set_column("K:K", 10)  # 设置K列的宽度
                '''-------------------------自适应列宽--------------------------------'''
                col = []
                for nnn in self.columns:
                    col.append(len(nnn.encode('gbk')))  # 转gbk格式计算列名长度
                # #  计算表头的字符宽度  # 其它格式不符合这套
                # column_widths = (
                #     a_pd.columns.to_series()
                #         .apply(lambda x: len(x.encode('gbk'))).values
                # )
                #  计算每列的最大字符宽度
                max_widths = (
                    a_pd.astype(str)
                        .applymap(lambda mb: len(mb.encode('gbk')))
                        .agg(max).values
                )
                # 计算整体最大宽度
                widths = np.max([col, max_widths], axis=0)
                for i, width in enumerate(widths):
                    writer1.set_column(i, i, width)
                '''-----------------------------------------------------------------'''
                writer.save()  # save file
                self.SendText.delete('1.0', END)
                txt = '保存成功, 请到result文件路径下查看!'
                self.SendText.insert(END, '{}\n\n'.format(txt))
                self.SendText.see(END)
                self.SendText.update()
        except Exception as exp:
            logging.info(exp)
            message = '保存失败, 界面暂无需要保存的数据. {}'.format(exp)
            self.SendText.delete('1.0', END)
            self.SendText.insert(END, '{}\n\n'.format(message))
            self.SendText.see(END)
            self.SendText.update()
            os.remove(writer)

    def save_result_cancel(self):
        t2 = threading.Thread(target=self.save_result, args=())
        t2.setDaemon(True)
        t2.start()

    # ---------------------------------------使用说明-----------------------------------------
    def instructions(self):
        self.SendText.delete('1.0', END)
        with open('config/instructions.txt', encoding='utf-8') as file:
            content = file.read()

        self.SendText.insert(END, '{}'.format(content))
        # SendText.see(END)
        self.SendText.update()

    def instructions_cancel(self):
        t2 = threading.Thread(target=self.instructions, args=())
        t2.setDaemon(True)
        t2.start()

    # ---------------------------------------语音播放-----------------------------------------
    def read(self):
        word_get = self.SendText.get('0.0', END)
        if len(word_get) == 1:
            word_get = '欢迎使用本软件, 请先点击使用说明, 或在文本框内输入你想听的话' \
                       'Welcome to use this software, please click the instructions first, ' \
                       'or enter what you want to hear in the text box'
        engine = pyttsx3.init()  # 对象创建

        # 输出本地语言包
        # voices = engine.getProperty('voices')
        # for voice in voices:
        #     print("Voice:")
        #     print(" - ID: %s" % voice.id)
        #     print(" - Name: %s" % voice.name)
        #     print(" - Languages: %s" % voice.languages)
        #     print(" - Gender: %s" % voice.gender)
        #     print(" - Age: %s" % voice.age)

        """ RATE"""
        # rate = engine.getProperty('rate')  # # 获取当前语速的详细信息
        # print(rate)  # 打印当前语音速率
        engine.setProperty('rate', 125)  # 设置新的语音速率

        """VOLUME"""
        # volume = engine.getProperty('volume')  # 了解当前音量水平（最小值 = 0 和最大值 = 1）
        # print(volume)  # 打印当前卷级别
        engine.setProperty('volume', 1.0)  # 将音量级别设置为 0 和 1 之间

        # """VOICE"""
        # -------------------微软语音包----------------------
        # # zh_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ZH-CN_HUIHUI_11.0"  # 中文包
        # zh_voice_id = r'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN - US_ZIRA_11.0'  # 英文包
        #
        # # 用语音包ID来配置engine
        # engine.setProperty('voice', zh_voice_id)

        # -------------------自带包--------------------------
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)  # 改变索引, 0 中英文, 1 英文

        engine.say(word_get)
        engine.runAndWait()
        engine.stop()

    def read_cancel(self):
        t2 = threading.Thread(target=self.read, args=())
        t2.setDaemon(True)
        t2.start()

    # ---------------------------------------清空界面-----------------------------------------
    def clear(self):
        """
        清空界面的显示结果
        :return: None
        """
        res = messagebox.askquestion('闪码科技  提示', '是否删除界面所有数据!')
        if res == 'yes':
            self.SendText.delete('1.0', END)
            self.table.delete(*self.table.get_children())
            self.progressbar.stop()

    def clear_cancel(self):
        t2 = threading.Thread(target=self.clear, args=())
        t2.setDaemon(True)
        t2.start()

    # ---------------------------------------选择检测文件夹-----------------------------------------
    def select_folder(self):
        """
        选择文件夹
        """
        global selected_folder
        selected_folder = filedialog.askdirectory(title='选择文件夹')  # 使用askdirectory函数选择文件夹
        self.select_path.set(selected_folder)
        if not selected_folder:
            message = '你已取消当前的文件选择操作!'
            tkinter.messagebox.showwarning(title='闪码科技  警告', message=message)
        else:
            self.SendText.delete('1.0', END)
            self.SendText.insert(END, '你选择的文件夹路径为: {}\n\n'.format(selected_folder))
            self.SendText.see(END)
            self.SendText.update()

    # ---------------------------------------选择模型-----------------------------------------
    def select_file(self):
        """
        选择模型
        """
        global selected_file_path
        # 使用askopenfilename函数选择单个文件
        selected_file_path = filedialog.askopenfilename(title='选择模型文件', filetypes=[('模型文件', '*.pt')])
        self.select_model_path.set(selected_file_path)
        if not selected_file_path:
            message = '你已取消当前的模型选择操作!'
            tkinter.messagebox.showwarning(title='闪码科技  警告', message=message)
        else:
            self.SendText.delete('1.0', END)
            self.SendText.insert(END, '你选择的模型路径为:\n\n{}\n\n'.format(selected_file_path))
            self.SendText.see(END)
            self.SendText.update()

    # ---------------------------------------初始化按钮-----------------------------------------
    # 加载相关参数，并初始化模型
    def model_init(self):
        global selected_file_path, device, model, imgsz, half, opt, colors, source, stride, names, save_txt
        try:
            # 模型相关参数配置
            parser = argparse.ArgumentParser()
            parser.add_argument('--weights', nargs='+', type=str, default=coco[0], help='model.pt path(s)')
            parser.add_argument('--source', type=str, default=coco[1], help='source')  # file/folder, 0 for webcam
            parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
            parser.add_argument('--conf-thres', type=float, default=coco[2], help='object confidence threshold')
            parser.add_argument('--iou-thres', type=float, default=coco[3], help='IOU threshold for NMS')
            parser.add_argument('--device', default=coco[4], help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
            parser.add_argument('--view-img', action='store_true', help='display results')
            parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
            parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
            parser.add_argument('--nosave', action='store_true', default=strtobool(coco[5]),
                                help='do not save images/videos')
            parser.add_argument('--classes', nargs='+', type=int, default=cla,
                                help='filter by class: --class 0, or --class 0 2 3')
            parser.add_argument('--agnostic-nms', action='store_true', default=True, help='class-agnostic NMS')
            parser.add_argument('--augment', action='store_true', help='augmented inference')
            parser.add_argument('--update', action='store_true', help='update all models')
            parser.add_argument('--project', default=coco[7], help='save results to project/name')
            parser.add_argument('--name', default=coco[8], help='save results to project/name')
            parser.add_argument('--exist-ok', action='store_true', default=strtobool(coco[6]),
                                help='existing project/name ok, do not increment')
            opt = parser.parse_args()
            # print(opt)
            # 默认使用opt中的设置（权重等）来对模型进行初始化
            source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
            # print(source, weights, view_img, save_txt, imgsz)
            # 若selected_file_path不为空，则使用此权重进行初始化
            if selected_file_path:
                weights = selected_file_path
                self.SendText.delete('1.0', END)
                self.SendText.insert(END, '你初始化的模型为:\n\n{}\n\n请耐心等待模型初始化完成的提示框!\n\n'.format(selected_file_path))
                self.SendText.see(END)
                self.SendText.update()
            else:
                self.SendText.delete('1.0', END)
                self.SendText.insert(END, '由于你未选择其它模型,将采用默认模型进行初始化:\n\n{}\n\n请耐心等待模型初始化完成的提示框!\n\n'.format(weights))
                self.SendText.see(END)
                self.SendText.update()
            device = select_device(opt.device)
            half = device.type != 'cpu'  # half precision only supported on CUDA
            cudnn.benchmark = True
            # Load model
            model = attempt_load(weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check img_size
            if half:
                model.half()  # to FP16

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
            tkinter.messagebox.showinfo(title='闪码科技  提示', message='模型初始化完成!')
        except Exception as exp:
            logging.info(exp)
            self.SendText.delete('1.0', END)
            self.SendText.insert(END, '发生了意料之外的错误, 请联系开发者进行解决! {}'.format(exp))
            self.SendText.see(END)
            self.SendText.update()

    def model_init_cancel(self):
        t2 = threading.Thread(target=self.model_init, args=())
        t2.setDaemon(True)
        t2.start()

    # ---------------------------------------检测按钮-----------------------------------------
    def detect(self):
        global device, model, imgsz, half, opt, colors, selected_folder, source, names, save_txt, path
        try:
            save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
            save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
            if selected_folder:
                source = selected_folder
                self.SendText.delete('1.0', END)
                self.SendText.insert(END, '开始检测!\n\n------------🚀🚀🚀🚀🚀🚀🚀-----------\n\n')
                self.SendText.see(END)
                self.SendText.update()
            else:
                self.SendText.delete('1.0', END)
                self.SendText.insert(END,
                                     '由于你未选择待检测的图片文件夹,将采用默认路径开始检测!\n\n------------🚀🚀🚀🚀🚀🚀🚀-----------\n\n'.format(
                                         source))
                self.SendText.see(END)
                self.SendText.update()

            # 加载数据
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            some = 0  # 控制进度条
            for path, img, im0s, vid_cap in dataset:
                # print(path, img, im0s, vid_cap)
                # print(path.split("\\")[-1])
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = model(img, augment=opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                           agnostic=opt.agnostic_nms)
                # Process detections
                ps = 0
                sm = 0
                rs = 0
                doupi = 0
                doujia = 0
                dougan = 0
                wj = 0
                yumi = 0
                th = 0
                zb = 0
                other = 0
                zhongyiji = 0
                caozi = 0
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # img.jpg

                    s += '%gx%g ' % img.shape[2:]  # print string
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        # print(s)
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = f'{names[int(cls)]} {conf:.2f}'
                            # print(label.split(' ')[0])
                            if label.split(' ')[0] == 'ps':
                                ps += 1
                            elif label.split(' ')[0] == 'sm':
                                sm += 1
                            elif label.split(' ')[0] == 'rs':
                                rs += 1
                            elif label.split(' ')[0] == 'doupi':
                                doupi += 1
                            elif label.split(' ')[0] == 'doujia':
                                doujia += 1
                            elif label.split(' ')[0] == 'dougan':
                                dougan += 1
                            elif label.split(' ')[0] == 'wj':
                                wj += 1
                            elif label.split(' ')[0] == 'yumi':
                                yumi += 1
                            elif label.split(' ')[0] == 'th':
                                th += 1
                            elif label.split(' ')[0] == 'zb':
                                zb += 1
                            elif label.split(' ')[0] == 'other':
                                other += 1
                            elif label.split(' ')[0] == 'zhongyiji':
                                zhongyiji += 1
                            else:
                                caozi += 1
                            if save_img:  # Add bbox to image
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                    together = sm + rs + zb + th + other + zhongyiji
                    zz = doupi + dougan + doujia + wj + yumi
                    gg = [path.split("\\")[-1], ps, together, zz, sm, rs, caozi, th, zb, zhongyiji, other, doupi,
                          doujia,
                          dougan, yumi, wj]
                    last = self.table.insert('', END, values=gg)  # 添加数据到末尾
                    self.table.see(last)  # 显示最后一行数据
                    self.table.update()
                    root.update()
                    # Print time (inference + NMS)
                    # print(f'{s}Done.')
                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                    some += 1
                self.progressbar['maximum'] = len(dataset)
                self.progressbar['value'] = some
                # time.sleep(0.1)
                root.update()
            self.SendText.insert(END, '检测完成!')
            self.SendText.see(END)
            self.SendText.update()
        except Exception as exp:
            logging.info(exp)
            self.SendText.delete('1.0', END)
            self.SendText.insert(END, '发生了意料之外的错误, 请联系开发者进行解决! {}'.format(exp))
            self.SendText.see(END)
            self.SendText.update()

    def detect_cancel(self):
        t2 = threading.Thread(target=self.detect, args=())
        t2.setDaemon(True)
        t2.start()

    # ------------------------------------------退出函数-----------------------------------
    def app_quit(self):
        global root
        root.destroy()  # 必须摧毁窗口 quit会无效一次
        root.quit()

    def app_quit_cancel(self):
        t2 = threading.Thread(target=self.app_quit, args=())
        t2.setDaemon(True)
        t2.start()

    # -----------------------------------------更多功能-------------------------------------
    def more_(self):
        self.SendText.delete('1.0', END)
        self.SendText.insert(END, '得加钱!')
        self.SendText.see(END)
        self.SendText.update()

    def create_window(self):
        # ---------------------------------------表格控件-----------------------------------------
        tabel_frame = tkinter.Frame(root)
        tabel_frame.pack(fill=BOTH, expand=True)
        xscroll = Scrollbar(tabel_frame, orient=HORIZONTAL)
        yscroll = Scrollbar(tabel_frame, orient=VERTICAL)
        self.columns = ['图片名称', '破碎粒', '总损伤粒', '杂质', '霉变粒', '热损粒', '草籽', '碳化粒', '紫斑粒', '种衣剂',
                        '其它损伤粒', '豆皮', '豆荚', '豆杆', '玉米', '无机杂质']
        self.table = ttk.Treeview(
            master=tabel_frame,  # 父容器
            height=15,  # 表格显示的行数,height行
            columns=self.columns,  # 显示的列
            show='headings',  # 隐藏首列
            xscrollcommand=xscroll.set,  # x轴滚动条
            yscrollcommand=yscroll.set,  # y轴滚动条
        )
        for column in self.columns:
            if column == '图片名称':
                self.table.heading(column=column, text=column, anchor=CENTER,
                                   command=lambda name=column:
                                   messagebox.showinfo('闪码科技  ', '检测文件夹路径下的{}'.format(name)))  # 定义表头
            else:
                self.table.heading(column=column, text=column, anchor=CENTER,
                                   command=lambda name=column:
                                   messagebox.showinfo('闪码科技  ', '图片蕴含的{}个数'.format(name)))  # 定义表头
            self.table.column(column=column, width=209, minwidth=90, anchor=CENTER)  # 定义列
            self.table.column(column, anchor=CENTER, width=90)
        #  x轴滚动条
        xscroll.config(command=self.table.xview)
        xscroll.pack(side=BOTTOM, fill=X)
        #  y轴滚动条
        self.table.configure(yscrollcommand=yscroll.set)  # 绑定y轴滚动条
        yscroll.config(command=self.table.yview)
        yscroll.pack(side=RIGHT, fill=Y)

        #  设置标题, 内容字体
        style = Style()
        style.configure("Treeview.Heading", font=('黑体', 12))  # 设置标题头的字体
        # style.configure("Treeview", font=(None, 11))  # 设置每一行的字体

        self.table.pack(fill=BOTH, expand=True, side='right')

        # # ---------------------------------------菜单栏-----------------------------------------

        menubar = Menu()
        root.config(menu=menubar)

        # create a File menu and add it to the menubar
        file_menu = Menu(menubar, tearoff=False)  # tearoff == True 有虚线
        help_menu = Menu(menubar, tearoff=False)

        # 给菜单命名
        menubar.add_cascade(label="更多", menu=file_menu)  # 这里的字体无法更改
        menubar.add_cascade(label="帮助", menu=help_menu)

        # 给菜单组件 add_command
        help_menu.add_command(label="语音播放", command=self.read_cancel, font=("楷体", 12))
        help_menu.add_command(label="闪码官网", command=sm_web, font=("楷体", 12))

        # 创建子菜单
        submenu = Menu(file_menu, tearoff=False)
        submenu.add_command(label="更多功能", font=("楷体", 12), command=self.more_)  # 只能更改这里的字体
        submenu.add_command(label="联系作者", font=("楷体", 12), command=self.more_)
        file_menu.add_cascade(label='详细介绍', menu=submenu, font=("楷体", 12, 'bold'))  # normal正常字体, bold加粗
        file_menu.add_separator()

        file_menu.add_command(label="需求反馈", font=("楷体", 12,), command=self.more_)
        file_menu.add_command(label="敬请期待", font=("楷体", 12,), command=self.more_)

        # ---------------------------------------操作区-----------------------------------------
        # 第一个框架，居左，y方向全部填充
        frame1 = Frame(root)
        # expand=1 当窗口改变可以填充
        frame1.pack(side="left", expand=0)
        btn_model = Button(frame1, text="选择模型", width=10, height=1, font=('', 11), command=self.select_file)
        CreateToolTip(btn_model, '选择视觉检测模型')
        btn_model.pack(side="top")
        btn_collection = Button(frame1, text="初始模型", width=10, height=1, font=('', 11), command=self.model_init_cancel)
        CreateToolTip(btn_collection, '初始视觉检测参数')
        btn_collection.pack(side="top", pady=15)
        btn_link = Button(frame1, text="选择文件", width=10, height=1, font=('', 11), command=self.select_folder)
        CreateToolTip(btn_link, '选择需要检测的文件夹, 不支持中文路径')
        btn_link.pack(side="top", pady=0)
        btn_start = Button(frame1, text="开始检测", width=10, height=1, font=('', 11), command=self.detect_cancel)
        CreateToolTip(btn_start, '开始对所选文件进行检测')
        btn_start.pack(side="top", pady=15)
        btn_use = Button(frame1, text="使用说明", width=10, height=1, font=('', 11), command=self.instructions_cancel)
        CreateToolTip(btn_use, '关于软件的使用方法')
        btn_use.pack(side="top", pady=0)
        btn_del = Button(frame1, text="删除某行", width=10, height=1, font=('', 11), command=self.del_some_cancel)
        CreateToolTip(btn_del, '点击误操作所在的数据行, 进行删除')
        btn_del.pack(side="top", pady=15)
        btn_save = Button(frame1, text="保存结果", width=10, height=1, font=('', 11), command=self.save_result_cancel)
        CreateToolTip(btn_save, '保存本次检测的数据')
        btn_save.pack(side="top", pady=0)

        # ---------------------------------------文本框-----------------------------------------
        frame3 = Frame(root)
        frame3.pack(side="left", anchor='n', padx=15, expand=1, fill="x")
        '''结果控件'''
        self.SendText = Text(frame3, width=78, height=14, undo=True)  # undo=True 开启撤销功能
        # photo = PhotoImage(file='config/icon.png')
        # SendText.image_create(END, image=photo)  # 用这个方法创建一个图片对象，并插入到“END”的位置
        self.SendText.pack(side="top", fill="x", expand=1)

        # Text内部支持复制，粘贴
        menubar = tkinter.Menu(root, tearoff=False)

        def cut(editor, event=None):
            editor.event_generate("<<Cut>>")

        def copy(editor, event=None):
            editor.event_generate("<<Copy>>")

        def paste(editor, event=None):
            editor.event_generate('<<Paste>>')

        def revoke(editor, event=None):
            editor.event_generate('<<Undo>>')

        def rightKey(event, editor):
            menubar.delete(0, 'end')
            menubar.add_command(label='剪切', accelerator='Ctrl+X', command=lambda: cut(editor))
            menubar.add_command(label='复制', accelerator='Ctrl+C', command=lambda: copy(editor))
            menubar.add_command(label='粘贴', accelerator='Ctrl+V', command=lambda: paste(editor))
            menubar.add_command(label='撤销', accelerator='Ctrl+Z', command=lambda: revoke(editor))
            menubar.post(event.x_root, event.y_root)

        # 绑定右键复制粘贴功能
        self.SendText.bind('<Button-3>', lambda x: rightKey(x, self.SendText))

        if cla is not None:
            self.SendText.delete('1.0', END)
            self.SendText.insert(END, '你将只检测这几类数据{}\n\n'.format(cla))
            self.SendText.see(END)
            self.SendText.update()

        frame2 = Frame(frame3)
        frame2.pack(side="top", anchor='n', expand=1, fill="x")
        label = tkinter.Label(frame2, text='检测进度: ', font=('', 11))
        label.pack(side="left", pady=19)
        self.progressbar = ttk.Progressbar(frame2)
        self.progressbar.pack(side="left", fill="x", pady=19, expand=1)
        # 设置进度条长度
        self.progressbar['length'] = 300
        # label1 = tkinter.Label(frame2, text='检测完成', font=('', 11))
        # label1.pack(side="left")

        # ---------------------------------------区块四-----------------------------------------
        frame4 = Frame(frame3)
        frame4.pack(side="bottom", fill='x', expand=1)
        btn_clear = Button(frame4, text="清空界面", width=10, height=1, font=('', 11), command=self.clear_cancel)
        CreateToolTip(btn_clear, '重置界面所有内容')
        btn_clear.pack(side="left", fill="x", expand=0)
        btn_quit = Button(frame4, text="退出程序", width=10, height=1, font=('', 11), command=self.app_quit_cancel)
        CreateToolTip(btn_quit, '注意保存数据!')
        btn_quit.pack(side="right", fill="x", expand=0)


def make_log():
    # ---------------------------------------日志------------------------------------------------
    os.makedirs('log', exist_ok=True)
    runtime = time.strftime("%m%d", time.localtime())
    # 配置日志文件
    logging.basicConfig(
        filename='log/machinedet' + '=' + runtime + '.log',  # 保存的文件名
        level=logging.INFO,
        datefmt='[%Y-%m-%d %H:%M:%S]',  # 日期格式
        format='%(asctime)s %(levelname)s %(filename)s [%(lineno)d] %(threadName)s : %(message)s',  # 保存数据格式
    )

    """
    # logging.debug('这个是调试时记录的日志信息')
    # logging.info('程序正常运行时记录的日志信息')
    # logging.warning('程序警告记录的信息')
    # logging.critical("特别严重的问题")
    # logging.error("程序错误时的记录，比如网络请求过慢等")
    """


def get_config():
    # ---------------------------------------读取配置文件-----------------------------------------
    filename = 'config/deploy.ini'
    config = configparser.ConfigParser()
    config.read(filename)

    # opt config
    items = config.items('Opt')
    coco = []
    for key in items:
        coco.append(key[1])

    cla = []
    for j in coco[9].split(' '):
        if j == 'None':
            cla = None
        else:
            cla.append(int(j))

    size = config.items('Resize')
    mn = []
    for si in size:
        mn.append(si[1])
    return coco, mn, cla


# -----------------------------------------------登录控件-------------------------------------------------
class Login(object):
    def __init__(self, master=None):
        # super().__init__(master)  # super()代表的是父类的定义 ，而不是父类的对像

        self.index = tkinter.Tk()  # 创建主窗口
        self.index.title('视觉检测登录界面')  # 设置主窗口标题

        # # 下面两行代码的作用是固定窗口大小，不可拉动调节
        # index.maxsize(500, 300)
        # index.minsize(500, 300)
        sw = self.index.winfo_screenwidth()
        sh = self.index.winfo_screenheight()
        ww = 500
        wh = 211
        x = (sw - ww) / 2
        y = (sh - wh) / 2
        self.index.geometry("%dx%d+%d+%d" % (ww, wh, x, y))
        '''窗口固定'''
        self.index.resizable(0, 0)
        self.index.iconphoto(False, PhotoImage(file='config/icon.png'))  # 设置logo
        # self.index.attributes("-toolwindow", True)  # 隐藏放大缩小按钮, 太丑
        self.user = {}  # 定义一个字典来存储用户的信息(key :账号 , value：密码)
        self.account = None
        self.password = None
        self.image_file = None
        self.new_login()

        self.index.mainloop()

    def new_login(self):
        # 加载图片
        canvas = tkinter.Canvas(self.index, width=500, height=300, bg=None)
        self.image_file = tkinter.PhotoImage(file="config/login.png")
        image = canvas.create_image(250, 0, anchor='n', image=self.image_file)
        canvas.pack()

        # 账号与密码文字标签
        account_lable = tkinter.Label(self.index, text='账号', bg='lightskyblue', fg='white', font=('Arial', 12), width=5,
                                      height=1)
        account_lable.place(relx=0.29, rely=0.3)
        pasw_lable = tkinter.Label(self.index, text='密码', bg='lightskyblue', fg='white', font=('Arial', 12), width=5,
                                   height=1)
        pasw_lable.place(relx=0.29, rely=0.5)

        # 身份初始值
        org_account = StringVar()
        org_account.set('jiangsu@shanma')
        # 密码初始值
        org_password = StringVar()
        org_password.set('sm@2023')

        # 账号与密码输入框
        self.account = tkinter.Entry(self.index, width=20, highlightthickness=1, highlightcolor='lightskyblue',
                                     relief='groove', textvariable=org_account)  # 账号输入框
        self.account.place(relx=0.4, rely=0.3)  # 添加进主页面,relx和rely意思是与父元件的相对位置
        self.password = tkinter.Entry(self.index, show='*', highlightthickness=1, highlightcolor='lightskyblue',
                                      relief='groove', textvariable=org_password)  # 密码输入框
        self.password.place(relx=0.4, rely=0.5)  # 添加进主页面

        with open('config/subscriber.txt', 'r') as f:
            for line in f.read().splitlines():
                if line == '':
                    continue
                self.user[line.split(",")[0]] = line.split(",")[1]

        # 登录与注册按钮
        loginBtn = tkinter.Button(self.index, text='登录', font=('宋体', 12), width=4, height=1, command=self.login,
                                  relief='solid',
                                  bd=0.5,
                                  bg='lightcyan')
        loginBtn.place(relx=0.41, rely=0.63)
        loginBtn1 = tkinter.Button(self.index, text='注册', font=('宋体', 12), width=4, height=1, bd=0.5,
                                   command=self.register_user,
                                   relief='solid',
                                   bg='lightcyan')
        CreateToolTip(loginBtn, '用户登录, 请先填写完整信息')
        CreateToolTip(loginBtn1, '用户注册, 欢迎使用')
        loginBtn1.place(relx=0.60, rely=0.63)
        self.index.bind('<Return>', self.login)  # 绑定enter键进入

    # 登录按钮处理函数
    def login(self, event=None):
        global root
        ac = self.account.get()
        ps = self.password.get()
        if ac == "" and ps == "":
            messagebox.showinfo("闪码科技  提示", "请先填写完整信息!")  # messagebox的方法
        elif self.user.get(ac) != ps:
            # self.account.delete(0, 'end')  # 清空文本框的内容
            # self.password.delete(0, 'end')  # 清空文本框的内容
            messagebox.showinfo("闪码科技  提示", "账号或者密码有误!")  # messagebox的方法
        else:
            try:
                # 摧毁登录界面
                self.index.destroy()
                self.index.quit()

                # ---------------------------------------主界面-----------------------------------------
                root = Tk()
                root.title('机器视觉检测程序')
                # root.wm_attributes('-transparentcolor', 'white')  # 设置透明界面
                '''得到屏幕宽度'''
                sw = root.winfo_screenwidth()
                '''得到屏幕高度'''
                sh = root.winfo_screenheight()
                ww = 648
                wh = 642
                x = (sw - ww) / 2
                y = (sh - wh) / 2
                root.geometry("%dx%d+%d+%d" % (ww, wh, x, y))
                root.resizable(mn[1], mn[0])
                root.iconphoto(False, PhotoImage(file='config/icon.png'))  # 设置logo
                Application(master=root)
                root.mainloop()
            except Exception as exp:
                logging.info(exp)

    def register_user(self):
        try:
            with open('config/subscriber.txt', 'r+') as f:
                username_info = self.account.get()
                password_info = self.password.get()
                username_good = True
                for line in f.read().splitlines():
                    if username_info == line.split(",")[0]:
                        username_good = False
                        break  # Stop the for loop from continuing
                if username_info == "" or password_info == "":
                    messagebox.showinfo("闪码科技  提示", "请先填写完整信息!")
                elif username_info in self.user:
                    messagebox.showinfo("闪码科技  提示", "该用户已存在!")
                elif username_good:
                    if len(password_info) < 6:
                        messagebox.showinfo("闪码科技  提示", "密码强度太弱, 为了您的账户安全, 请设置至少6位数密码!")
                    else:
                        f.write('\n' + username_info + "," + password_info + "\n")
                        self.user[username_info] = password_info
                        self.account.delete(0, END)
                        self.password.delete(0, END)
                        messagebox.showinfo("闪码科技  提示  ", "注册成功!")
                else:
                    logging.info('注册发生了预期之外的错误!')
                    messagebox.showinfo("闪码科技  提示", "发生了意料之外的错误!!")
        except Exception as exp:
            logging.info(exp)


if __name__ == '__main__':
    try:
        make_log()  # 日志
        coco, mn, cla = get_config()  # 主程序配置
        index = Login()  # 登录界面
    except Exception as e:
        logging.info(e)
