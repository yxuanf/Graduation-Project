import os
import shutil
import sys
import warnings
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from genie2pgm.attribute import Attributes
from genie2pgm.noisymax import NoisyMax
from genie2pgm.simplemodel import SimpleDiscreteModel
from genie2pgmpy_ui import Ui_Form
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import ApproxInference
from pgmpy.inference import VariableElimination


class window(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.chooseNode = None
        self.axes = None
        self.canvas = None
        self.virtual_cpt = []
        self.querymode = "精确推理"
        self.sampleNum.setVisible(False)
        self.page1.clicked.connect(self.page1_clicked)
        self.page2.clicked.connect(self.page2_clicked)
        self.openFile.clicked.connect(self.openFile_clicked)
        self.showBN.clicked.connect(self.showBN_clicked)
        self.closeBN.clicked.connect(self.clear_clicked)
        self.savebutton.clicked.connect(self.savemodel_clicked)
        self.cptvariable.activated.connect(self.cptvariable_selected)
        self.queryCombobox.activated.connect(self.queryCombobox_selected)
        self.queryMode.activated.connect(self.queryMode_selected)
        self.clearButton.clicked.connect(self.clearButton_clicked)
        self.sampleNum.returnPressed.connect(self.get_SampleNum)
        self.querybutton.clicked.connect(self.query_button_clicked)
        self.stackedWidget.setCurrentIndex(0)
        self.palette1 = self.page1.styleSheet()
        self.palette2 = self.page2.styleSheet()
        self.red = QColor(255, 0, 0)
        self.white = QColor(255, 255, 255)

    def page1_clicked(self):
        self.stackedWidget.setCurrentIndex(0)
        self.page1.setStyleSheet(self.palette1)
        self.page2.setStyleSheet(self.palette2)

    def page2_clicked(self):
        self.stackedWidget.setCurrentIndex(1)
        self.page1.setStyleSheet(self.palette2)
        self.page2.setStyleSheet(self.palette1)

    # 打开xml或者xdsl文件

    def openFile_clicked(self):
        self.path, _ = QFileDialog.getOpenFileName(
            self, "读取xml文件", "", "(*.xml *.xdsl)"
        )
        if self.path:
            self.path, self.filename = change_file_extension(self.path)
            print(self.path)
            self.judge_model(self.path)
            self.showBN.setVisible(True)
            self.closeBN.setVisible(True)
            self.model = self.module.model
            self.nodes = list(self.model.nodes())
            self.edges = self.model.edges()
            self.statename = self.module.state_names
            self.nodecard = self.module.evidence_card
            self.menu = QMenu(self.evidenceButton)
            self.clearButton_clicked()
        try:
            if isinstance(self.module, NoisyMax):
                self.module.add_noisymax_cpd()
            elif isinstance(self.module, SimpleDiscreteModel):
                self.module.add_cpd(self.model, self.module.getcpd())
            if self.model.check_model():
                self.infer = None
                QMessageBox.information(QWidget(), "导入成功", "成功添加先验概率!")
                self.cptvariable.setVisible(True)
                self.CPTlabel.setVisible(True)
                self.showCPT.setHidden(False)
                self.cptvariable.clear()
                self.cptvariable.addItems(self.nodes)
                self.queryCombobox.clear()
                self.queryCombobox.addItems(self.nodes)
                # 增加证据菜单
                self.addMenu()
            else:
                QMessageBox.critical(
                    QWidget(),
                    "错误！",
                    "无法正确导入模型的先验概率!,请检查导入模型的正确性！",
                )
                return
        except AttributeError:
            QMessageBox.critical(
                QWidget(), "错误！", "请在导入先验概率前打开Genie相关文件!"
            )

    def showBN_clicked(self):
        try:
            self.EdgesBrowser.clear()
            self.EdgesBrowser.append(str(self.module.model))
            for edge in self.edges:
                self.EdgesBrowser.append(str(edge))
        except AttributeError:
            QMessageBox.critical(
                QWidget(), "错误！", "请在展示贝叶斯网络前打开Genie相关文件!"
            )

    # 保存模型
    def savemodel_clicked(self):
        try:
            informtion = Attributes(self.path)
            self.model.save(f"./model/{self.filename}.bif", filetype="bif")
            informtion.getInformation()
            QMessageBox.information(QWidget(), "保存成功", "已成功保存模型！")
        except:
            QMessageBox.critical(
                QWidget(), "错误！", "请在保存模型前打开Genie相关文件!"
            )

    def addMenu(self):
        # 增加菜单
        self.menu = QMenu(self.evidenceButton)
        for node in self.nodes:
            subMeau = QMenu(node, self.menu)
            for state in self.statename[node]:
                action = QAction(state, subMeau)
                action.setCheckable(True)
                action.triggered.connect(
                    lambda _, state=node, substate=state: self.handle_Menu_action(
                        state, substate
                    )
                )
                subMeau.addAction(action)
            virtual_action = QAction("virtual evidence", subMeau)
            virtual_action.setCheckable(True)
            virtual_action.triggered.connect(
                lambda _, node=node: self.handle_Virtual_action(node)
            )
            # 分割线
            subMeau.addSeparator()
            subMeau.addAction(virtual_action)
            self.menu.addMenu(subMeau)
            self.menu.setStyleSheet(
                "QMenu {background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);selection-color: rgb(255, 0, 0);}"
            )
        self.evidenceButton.setMenu(self.menu)
        self.evidence = {}

    def clear_clicked(self):
        self.EdgesBrowser.clear()

    def clearButton_clicked(self):
        self.evidence = {}
        self.virtual_cpt = []
        self.showEvidence.clear()
        self.queryCombobox.clear()
        self.addMenu()
        # 重新加载查询节点
        self.queryCombobox.addItems(self.nodes)

    def cptvariable_selected(self):
        temp = sys.stdout
        sys.stdout = EmittingStr(textWritten=self.QTextBrowser_output)
        self.showCPT.clear()
        self.chooseNode = self.cptvariable.currentText()
        for i in self.model.get_cpds():
            if i.variable == self.chooseNode:
                cpt = i
        print(cpt)
        sys.stdout = temp

    def handle_Menu_action(self, state, substate):
        self.showEvidence.clear()
        self.evidence[state] = substate
        for node, state in self.evidence.items():
            index = self.queryCombobox.findText(node)
            self.queryCombobox.removeItem(index)
            text = f"{node}---------------->{state}"
            self.showEvidence.append(text)

    def handle_Virtual_action(self, node):
        self.virtual_win = QDialog()
        self.virtual_win.setWindowTitle("Set Virtual Evidence")
        layout = QVBoxLayout()
        self.virtual_prob = []
        font = QFont()
        font.setPointSize(12)
        font.setFamily("Microsoft YaHei UI")
        for state in self.statename[node]:
            h_layout = QHBoxLayout()
            label = QLabel(state + ":", self)
            label.setFont(font)
            h_layout.addWidget(label)
            line_edit = QLineEdit(self)
            line_edit.setFont(font)
            line_edit.setText(str(1 / len(self.statename[node])))
            h_layout.addWidget(line_edit)
            self.virtual_prob.append(line_edit)
            layout.addLayout(h_layout)
        button = QPushButton("输入完成", self)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button.setFont(font)
        layout.addWidget(button)
        button.clicked.connect(
            lambda _, variable=node: self.prob_button_clicked(variable)
        )
        self.virtual_win.setLayout(layout)
        self.virtual_win.show()

    def prob_button_clicked(self, variable):
        probabilities = []
        statename = {}
        statename[variable] = self.statename[variable]
        variable_card = self.nodecard[variable]
        total_probability = 0
        for line_edit in self.virtual_prob:
            probability = float(line_edit.text() or 0)
            if probability < 0 or probability > 1:
                QMessageBox.critical(
                    QWidget(), "Error", "Probability value must be between 0 and 1."
                )
                return
            probabilities.append([probability])
            total_probability += probability
        if total_probability != 1:
            QMessageBox.critical(
                QWidget(), "Error", "The sum of probabilities must be 1."
            )
            return
        for i in range(variable_card):
            text = f"{variable}({self.statename[variable][i]})---------------->{probabilities[i][0]}"
            self.showEvidence.append(text)
        cpt = TabularCPD(
            variable=variable,
            variable_card=variable_card,
            values=probabilities,
            state_names=statename,
        )
        self.virtual_cpt.append(cpt)
        self.virtual_win.accept()

    def queryCombobox_selected(self):
        self.variable = [self.queryCombobox.currentText()]

    def queryMode_selected(self):
        self.querymode = self.queryMode.currentText()
        if self.querymode == "近似采样推理":
            self.sampleNum.setVisible(True)
        else:
            self.sampleNum.setVisible(False)

    def query_button_clicked(self):
        try:
            if self.querymode == "精确推理":
                if self.canvas != None:
                    self.picLayout.removeWidget(self.canvas)
                self.sampleNum.setVisible(False)
                self.infer = VariableElimination(model=self.model)
                self.query = self.infer.query(
                    variables=self.variable,
                    evidence=self.evidence,
                    virtual_evidence=self.virtual_cpt,
                )
                print(self.query)
                self.showpic(self.query)
            elif self.querymode == "近似采样推理":
                self.infer = ApproxInference(model=self.model)
                self.query = self.infer.query(
                    variables=self.variable,
                    evidence=self.evidence,
                    n_samples=self.count,
                    virtual_evidence=self.virtual_cpt,
                )
                if self.canvas != None:
                    self.picLayout.removeWidget(self.canvas)
                self.showpic(self.query)
        except AttributeError:
            QMessageBox.critical(QWidget(), "Error", "请选择需要查询的节点！")

    def get_SampleNum(self):
        try:
            text = self.sampleNum.text()
            self.count = int(text)
        except ValueError:
            QMessageBox.critical(QWidget(), "错误！", "请输入正确的采样数量！")

    # 判定是否为nosiyMax模型
    def judge_model(self, path):
        nodes = ET.parse(path).getroot().find("nodes")
        #
        if not nodes.findall("noisymax"):
            self.module = SimpleDiscreteModel(self.path)
        else:
            self.module = NoisyMax(self.path)
            QMessageBox.information(
                QWidget(), "Model type", "网络中存在NoisyMax类型节点!"
            )
            return

    def showpic(self, query):
        # seaborn样式
        sns.set(palette="muted", color_codes=True)
        # 解决Seaborn中文显示问题
        sns.set(font="Microsoft YaHei", font_scale=0.8)
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
        # 解决无法显示符号的问题
        plt.rcParams["axes.unicode_minus"] = False
        # get probability
        data = query.values
        # state = self.statename[query.variables[0]]
        state = query.state_names[query.variables[0]]
        print(state, data)
        print(query)
        self.fig, self.axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
        plt.rcParams["font.size"] = 12
        self.fig.suptitle(
            f"{query.variables[0]}", font={"family": "Microsoft YaHei", "size": 12}
        )
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.axes[0].bar(state, data, color=colors[: len(data)])
        self.axes[0].tick_params(axis="x", labelsize=12)
        self.axes[0].tick_params(axis="y", labelsize=12)
        for i, j in enumerate(data):
            self.axes[0].text(i, 1.01 * j, str(round(j, 4)), ha="center", va="bottom")
        _, l_text, p_text = self.axes[1].pie(data, labels=state, autopct="%.4f%%")
        for t in p_text:
            t.set_size(16)
        for t in l_text:
            t.set_size(16)
        self.axes[1].legend(prop={"size": 12})
        self.axes[1].tick_params(axis="x", labelsize=12)
        plt.tight_layout()
        self.canvas = FigureCanvas(self.fig)
        self.picLayout.addWidget(self.canvas)

    # 作为槽函数于Stream的信号连接, 内部参数于信号发射参数相同
    def QTextBrowser_output(self, text):
        cursor = self.showCPT.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.showCPT.setTextCursor(cursor)
        self.showCPT.ensureCursorVisible()


def change_file_extension(path, new_extension=".xml"):
    file_name = os.path.splitext(path)[0]
    old_extension = os.path.splitext(path)[-1]
    if old_extension.replace(".", "") == "xdsl":
        shutil.copy2(path, file_name + new_extension)
    return path, file_name.split("/")[-1]


# 自定义信号


class EmittingStr(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))
        QApplication.processEvents()

    # RuntimeError: wrapped C/C++ object of type EmittingStr has been deleted
    def flush(self):
        pass


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)
    myWin = window()
    myWin.setWindowTitle("Genie贝叶斯网络工具箱")
    myWin.show()
    msg_box = QMessageBox.warning(
        QWidget(), "温馨提示", "请在使用该工具之前导入Xml文件"
    )
    sys.exit(app.exec_())
