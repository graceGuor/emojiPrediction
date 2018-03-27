# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 19:42:37 2018

@author: ly
"""

from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams
from pdfminer.pdfparser import PDFParser,PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
import pdfminer.pdfinterp

path = r'C:\Users\ly\Desktop\新建文件夹\LIWC2015 dictionary poster.pdf'
output = r'C:\Users\ly\Desktop\新建文件夹\output.txt'
fp = open(path,"rb")
parser=PDFParser(fp)
#PDF文档对象
doc = PDFDocument()
parser.set_document(doc)
doc.set_parser(parser)

doc.initialize("")

#创建pdf资源管理器
resource = PDFResourceManager()

#参数分析器
laparam = LAParams()

#创建一个聚合器
device = PDFPageAggregator(resource,laparams=laparam)

#创建PDF页面解释器
interpreter=PDFPageInterpreter(resource,device)

#使用文档对象得到页面的集合
with open(output,'w') as f2:
    for page in doc.get_pages():
        #使用页面解释器来读取
        interpreter.process_page(page)
    
        #使用聚合器来获取内容
        layout=  device.get_result()
    
        for out in layout:
            if hasattr(out,"get_text"):
                f2.write(out.get_text())
