# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\NUDT\code\python\Genie1_2\GUITest\genie2pgmpy.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(929, 672)
        Form.setStyleSheet(
            "background-color: qlineargradient(x0:0, y1:0,x0:1, y2:1, stop:0 rgb(20, 32, 44), stop:1 rgb(37, 85, 117));\n"
            "border-radius:20px  "
        )
        self.gridLayout_4 = QtWidgets.QGridLayout(Form)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.verticalLayout.addItem(spacerItem)
        self.page1 = QtWidgets.QPushButton(Form)
        self.page1.setStyleSheet(
            "QPushButton{\n"
            "selection-color: rgb(255, 0, 0);\n"
            "border-radius:6px;\n"
            "color: rgb(255, 0,0);\n"
            "border-style:none;\n"
            "background-color: rgb(255,255,255);\n"
            'font: 16pt "楷体";\n'
            "padding:6px;\n"
            "}\n"
            ""
        )
        self.page1.setObjectName("page1")
        self.verticalLayout.addWidget(self.page1)
        self.page2 = QtWidgets.QPushButton(Form)
        self.page2.setStyleSheet(
            "QPushButton{\n"
            "selection-color: rgb(255, 0, 0);\n"
            "border-radius:6px;\n"
            "color: rgb(255, 255, 255);\n"
            "border-style:none;\n"
            "background-color: transparent;\n"
            'font: 16pt "楷体";\n'
            "padding:6px;\n"
            "}\n"
            ""
        )
        self.page2.setObjectName("page2")
        self.verticalLayout.addWidget(self.page2)
        self.savebutton = QtWidgets.QPushButton(Form)
        self.savebutton.setStyleSheet(
            "QPushButton{\n"
            "selection-color: rgb(255, 0, 0);\n"
            "border-radius:6px;\n"
            "color: rgb(255, 255, 255);\n"
            "border-style:none;\n"
            "background-color: transparent;\n"
            'font: 16pt "楷体";\n'
            "padding:6px;\n"
            "}\n"
            "QPushButton:hover{\n"
            "    color: rgb(255, 0, 0);\n"
            "    background-color: rgb(255,255,255);\n"
            "}"
        )
        self.savebutton.setObjectName("savebutton")
        self.verticalLayout.addWidget(self.savebutton)
        spacerItem1 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.verticalLayout.addItem(spacerItem1)
        self.verticalLayout.setStretch(0, 2)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 1)
        self.verticalLayout.setStretch(3, 1)
        self.verticalLayout.setStretch(4, 2)
        self.horizontalLayout_5.addLayout(self.verticalLayout)
        self.stackedWidget = QtWidgets.QStackedWidget(Form)
        self.stackedWidget.setStyleSheet(
            "background-color: \n"
            "qlineargradient(x0:0, \n"
            "y1:0,\n"
            "x0:1, \n"
            "y2:1, \n"
            "stop:0 rgb(20, 32, 44), \n"
            "stop:1 rgb(37, 85, 117));\n"
            "border-radius:20px  "
        )
        self.stackedWidget.setObjectName("stackedWidget")
        self.modelStructure = QtWidgets.QWidget()
        self.modelStructure.setObjectName("modelStructure")
        self.gridLayout = QtWidgets.QGridLayout(self.modelStructure)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout_page1 = QtWidgets.QGridLayout()
        self.gridLayout_page1.setObjectName("gridLayout_page1")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        spacerItem2 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_2.addItem(spacerItem2, 0, 3, 1, 1)
        self.openFile = QtWidgets.QPushButton(self.modelStructure)
        self.openFile.setStyleSheet(
            "QPushButton{\n"
            "padding:8px;\n"
            "    color:rgb(0,0,0);\n"
            "    background-color:rgb(226, 230, 255);\n"
            '    font: 12pt "楷体";\n'
            "    border-radius:16px;\n"
            "}\n"
            "QPushButton:hover{\n"
            "    color: rgb(0, 0, 0);\n"
            "    background-color: rgb(43, 162, 239);\n"
            "}\n"
            ""
        )
        self.openFile.setObjectName("openFile")
        self.gridLayout_2.addWidget(self.openFile, 0, 0, 1, 1)
        self.showBN = QtWidgets.QPushButton(self.modelStructure)
        self.showBN.setStyleSheet(
            "QPushButton{\n"
            "padding:8px;\n"
            "    color:rgb(0,0,0);\n"
            "    background-color:rgb(226, 230, 255);\n"
            '    font: 12pt "楷体";\n'
            "    border-radius:16px;\n"
            "}\n"
            "QPushButton:hover{\n"
            "    color: rgb(0, 0, 0);\n"
            "    background-color: rgb(43, 162, 239);\n"
            "}\n"
            ""
        )
        self.showBN.setObjectName("showBN")
        self.gridLayout_2.addWidget(self.showBN, 0, 1, 1, 1)
        self.closeBN = QtWidgets.QPushButton(self.modelStructure)
        self.closeBN.setStyleSheet(
            "QPushButton{\n"
            "padding:8px;\n"
            "    color:rgb(0,0,0);\n"
            "    background-color:rgb(226, 230, 255);\n"
            '    font: 12pt "楷体";\n'
            "    border-radius:16px;\n"
            "}\n"
            "QPushButton:hover{\n"
            "    color: rgb(0, 0, 0);\n"
            "    background-color: rgb(43, 162, 239);\n"
            "}\n"
            ""
        )
        self.closeBN.setObjectName("closeBN")
        self.gridLayout_2.addWidget(self.closeBN, 0, 4, 1, 1)
        self.gridLayout_2.setColumnStretch(0, 1)
        self.gridLayout_2.setColumnStretch(1, 1)
        self.gridLayout_2.setColumnStretch(3, 7)
        self.gridLayout_2.setColumnStretch(4, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 0, 1, 1, 1)
        self.gridLayout_7.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.EdgesBrowser = QtWidgets.QTextBrowser(self.modelStructure)
        self.EdgesBrowser.setStyleSheet(
            "color: rgb(0, 0, 0);\n"
            'font: 12pt "楷体";\n'
            "background-color:qlineargradient(spread:reflect, x1:0.0166818, y1:0, x2:1, y2:1, stop:0 rgba(255, 255, 255, 255), stop:0.931818 rgba(60, 88, 112, 255));\n"
            "border:0px solid red;\n"
            "border-radius:12px\n"
            "\n"
            ""
        )
        self.EdgesBrowser.setObjectName("EdgesBrowser")
        self.gridLayout_7.addWidget(self.EdgesBrowser, 1, 0, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_7, 1, 0, 1, 1)
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.BN_structure = QtWidgets.QLabel(self.modelStructure)
        self.BN_structure.setStyleSheet(
            'font: 20pt "楷体";\n'
            "border:none;\n"
            "background-color: rgba(255, 255, 255, 0);\n"
            "color: rgb(255, 255, 255);\n"
            ""
        )
        self.BN_structure.setObjectName("BN_structure")
        self.gridLayout_6.addWidget(self.BN_structure, 0, 1, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_6.addItem(spacerItem3, 0, 0, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_6.addItem(spacerItem4, 0, 2, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_6, 0, 0, 1, 1)
        self.gridLayout_page1.addLayout(self.gridLayout_5, 0, 0, 1, 1)
        self.gridLayout_8 = QtWidgets.QGridLayout()
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.gridLayout_9 = QtWidgets.QGridLayout()
        self.gridLayout_9.setObjectName("gridLayout_9")
        spacerItem5 = QtWidgets.QSpacerItem(
            298, 17, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_9.addItem(spacerItem5, 0, 2, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(
            298, 17, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_9.addItem(spacerItem6, 0, 0, 1, 1)
        self.CPTlabel = QtWidgets.QLabel(self.modelStructure)
        self.CPTlabel.setStyleSheet(
            'font: 20pt "楷体";\n'
            "border:none;\n"
            "background-color: rgba(255, 255, 255, 0);\n"
            "color: rgb(255, 255, 255);"
        )
        self.CPTlabel.setObjectName("CPTlabel")
        self.gridLayout_9.addWidget(self.CPTlabel, 0, 1, 1, 1)
        self.gridLayout_8.addLayout(self.gridLayout_9, 0, 0, 1, 1)
        self.gridLayout_10 = QtWidgets.QGridLayout()
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.cptvariable = QtWidgets.QComboBox(self.modelStructure)
        self.cptvariable.setStyleSheet(
            'font: 12pt "楷体";\n'
            "background-color: rgb(255, 255, 255);\n"
            "color: rgb(0, 0, 0);\n"
            "selection-color: rgb(255, 0, 0);"
        )
        self.cptvariable.setObjectName("cptvariable")
        self.horizontalLayout_2.addWidget(self.cptvariable)
        spacerItem7 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout_2.addItem(spacerItem7)
        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 3)
        self.gridLayout_10.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        self.showCPT = QtWidgets.QTextBrowser(self.modelStructure)
        self.showCPT.setStyleSheet(
            "color: rgb(0, 0, 0);\n"
            'font: 12pt "楷体";\n'
            "background-color:qlineargradient(spread:reflect, x1:0.0166818, y1:0, x2:1, y2:1, stop:0 rgba(255, 255, 255, 255), stop:0.931818 rgba(60, 88, 112, 255));\n"
            "border:0px solid red;\n"
            "border-radius:12px\n"
            ""
        )
        self.showCPT.setObjectName("showCPT")
        self.gridLayout_10.addWidget(self.showCPT, 1, 0, 1, 1)
        self.gridLayout_8.addLayout(self.gridLayout_10, 1, 0, 1, 1)
        self.gridLayout_page1.addLayout(self.gridLayout_8, 1, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_page1, 0, 0, 1, 1)
        self.stackedWidget.addWidget(self.modelStructure)
        self.Query = QtWidgets.QWidget()
        self.Query.setObjectName("Query")
        self.gridLayout_27 = QtWidgets.QGridLayout(self.Query)
        self.gridLayout_27.setObjectName("gridLayout_27")
        self.gridLayout_25 = QtWidgets.QGridLayout()
        self.gridLayout_25.setObjectName("gridLayout_25")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.evidenceButton = QtWidgets.QPushButton(self.Query)
        self.evidenceButton.setStyleSheet(
            "QPushButton{\n"
            "    selection-color: rgb(0, 0, 255);\n"
            "    padding:8px;\n"
            "    color:rgb(0,0,0);\n"
            "    background-color:rgb(226, 230, 255);\n"
            '    font: 12pt "楷体";\n'
            "    border-radius:16px;\n"
            "}\n"
            "\n"
            ""
        )
        self.evidenceButton.setObjectName("evidenceButton")
        self.horizontalLayout_3.addWidget(self.evidenceButton)
        self.clearButton = QtWidgets.QPushButton(self.Query)
        self.clearButton.setStyleSheet(
            "QPushButton{\n"
            "padding:8px;\n"
            "    color:rgb(0,0,0);\n"
            "    background-color:rgb(226, 230, 255);\n"
            '    font: 12pt "楷体";\n'
            "    border-radius:16px;\n"
            "}\n"
            "QPushButton:hover{\n"
            "    color: rgb(0, 0, 0);\n"
            "    background-color: rgb(43, 162, 239);\n"
            "}\n"
            ""
        )
        self.clearButton.setObjectName("clearButton")
        self.horizontalLayout_3.addWidget(self.clearButton)
        spacerItem8 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout_3.addItem(spacerItem8)
        self.sampleNum = QtWidgets.QLineEdit(self.Query)
        self.sampleNum.setStyleSheet(
            "background-color: rgb(255, 255, 255);\n"
            "color: rgb(0, 0, 0);\n"
            'font: 12pt "Times New Roman";'
        )
        self.sampleNum.setObjectName("sampleNum")
        self.horizontalLayout_3.addWidget(self.sampleNum)
        self.queryMode = QtWidgets.QComboBox(self.Query)
        self.queryMode.setStyleSheet(
            "background-color: rgb(255,255, 255);\n"
            'font: 12pt "Microsoft YaHei UI";\n'
            "selection-color: rgb(255, 0, 0);"
        )
        self.queryMode.setObjectName("queryMode")
        self.queryMode.addItem("")
        self.queryMode.addItem("")
        self.horizontalLayout_3.addWidget(self.queryMode)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 1)
        self.horizontalLayout_3.setStretch(2, 9)
        self.horizontalLayout_3.setStretch(3, 2)
        self.horizontalLayout_3.setStretch(4, 4)
        self.verticalLayout_4.addLayout(self.horizontalLayout_3)
        self.showEvidence = QtWidgets.QTextBrowser(self.Query)
        self.showEvidence.setStyleSheet(
            "QFrame{\n"
            "    background-color:qlineargradient(spread:reflect, x1:0.0166818, y1:0, x2:1, y2:1, stop:0 rgba(255, 255, 255, 255), stop:0.931818 rgba(60, 88, 112, 255));\n"
            "    border:0px solid red;\n"
            "    color: rgb(0, 0,0);\n"
            '    font: 12pt "楷体";\n'
            "border-radius:12px\n"
            "}"
        )
        self.showEvidence.setObjectName("showEvidence")
        self.verticalLayout_4.addWidget(self.showEvidence)
        self.verticalLayout_4.setStretch(1, 2)
        self.gridLayout_25.addLayout(self.verticalLayout_4, 0, 0, 1, 1)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.gridLayout_12 = QtWidgets.QGridLayout()
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.gridLayout_24 = QtWidgets.QGridLayout()
        self.gridLayout_24.setObjectName("gridLayout_24")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.queryCombobox = QtWidgets.QComboBox(self.Query)
        self.queryCombobox.setStyleSheet(
            'font: 12pt "楷体";\n'
            "color: rgb(0, 0, 0);\n"
            "selection-color: rgb(255, 0, 0);\n"
            "background-color: rgb(255, 255, 255);"
        )
        self.queryCombobox.setObjectName("queryCombobox")
        self.horizontalLayout.addWidget(self.queryCombobox)
        self.querybutton = QtWidgets.QPushButton(self.Query)
        self.querybutton.setStyleSheet(
            "QPushButton{\n"
            "padding:8px;\n"
            "    color:rgb(0,0,0);\n"
            "    background-color:rgb(226, 230, 255);\n"
            '    font: 12pt "楷体";\n'
            "    border-radius:16px;\n"
            "}\n"
            "QPushButton:hover{\n"
            "    color: rgb(0, 0, 0);\n"
            "    background-color: rgb(43, 162, 239);\n"
            "}\n"
            ""
        )
        self.querybutton.setObjectName("querybutton")
        self.horizontalLayout.addWidget(self.querybutton)
        spacerItem9 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout.addItem(spacerItem9)
        self.horizontalLayout.setStretch(0, 2)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(2, 8)
        self.gridLayout_24.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.gridLayout_12.addLayout(self.gridLayout_24, 0, 0, 1, 1)
        self.verticalLayout_5.addLayout(self.gridLayout_12)
        self.picLayout = QtWidgets.QGridLayout()
        self.picLayout.setObjectName("picLayout")
        self.frame = QtWidgets.QFrame(self.Query)
        self.frame.setStyleSheet(
            "QFrame{\n"
            "    background-color:qlineargradient(spread:reflect, x1:0.0166818, y1:0, x2:1, y2:1, stop:0 rgba(255, 255, 255, 255), stop:0.931818 rgba(60, 88, 112, 255));\n"
            "    border:0px solid red;\n"
            "    color: rgb(255, 254, 255);\n"
            "border-radius:12px\n"
            "}"
        )
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.picLayout.addWidget(self.frame, 0, 0, 1, 1)
        self.verticalLayout_5.addLayout(self.picLayout)
        self.verticalLayout_5.setStretch(0, 1)
        self.verticalLayout_5.setStretch(1, 7)
        self.gridLayout_25.addLayout(self.verticalLayout_5, 1, 0, 1, 1)
        self.gridLayout_25.setRowStretch(0, 1)
        self.gridLayout_25.setRowStretch(1, 4)
        self.gridLayout_27.addLayout(self.gridLayout_25, 0, 0, 1, 1)
        self.stackedWidget.addWidget(self.Query)
        self.horizontalLayout_5.addWidget(self.stackedWidget)
        self.horizontalLayout_5.setStretch(0, 1)
        self.horizontalLayout_5.setStretch(1, 8)
        self.gridLayout_4.addLayout(self.horizontalLayout_5, 0, 0, 1, 1)

        self.retranslateUi(Form)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.page1.setText(_translate("Form", "构建网络"))
        self.page2.setText(_translate("Form", "模型推理"))
        self.savebutton.setText(_translate("Form", "导出模型"))
        self.openFile.setText(_translate("Form", "导入"))
        self.showBN.setText(_translate("Form", "展示"))
        self.closeBN.setText(_translate("Form", "清除"))
        self.BN_structure.setText(_translate("Form", "贝叶斯网络结构"))
        self.CPTlabel.setText(_translate("Form", "概率表"))
        self.evidenceButton.setText(_translate("Form", "设置证据"))
        self.clearButton.setText(_translate("Form", "清除证据"))
        self.queryMode.setItemText(0, _translate("Form", "精确推理"))
        self.queryMode.setItemText(1, _translate("Form", "近似采样推理"))
        self.querybutton.setText(_translate("Form", "推理"))
