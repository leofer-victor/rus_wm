# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'deepusnav.ui'
##
## Created by: Qt User Interface Compiler version 6.11.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QGraphicsView, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QSizePolicy,
    QSpacerItem, QSplitter, QVBoxLayout, QWidget)

class Ui_DeepUSNavConsole(object):
    def setupUi(self, DeepUSNavConsole):
        if not DeepUSNavConsole.objectName():
            DeepUSNavConsole.setObjectName(u"DeepUSNavConsole")
        DeepUSNavConsole.resize(1500, 900)
        DeepUSNavConsole.setMinimumSize(QSize(1200, 760))
        self.centralwidget = QWidget(DeepUSNavConsole)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setStyleSheet(u"* { background-color: #25272b; color: #e7e7e7; font-size: 12pt; }\n"
"QGroupBox { border: 1px solid #59606a; border-radius: 4px; margin-top: 12px; padding-top: 10px; font-weight: bold; }\n"
"QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }\n"
"QPushButton { background-color: #41464f; border: 1px solid #69717d; border-radius: 3px; padding: 7px; }\n"
"QPushButton:hover { background-color: #515965; }\n"
"QPushButton:pressed { background-color: #30353c; }\n"
"QPushButton:disabled { color: #777; background-color: #303236; }\n"
"QPushButton#stop_robot { background-color: #9d3434; font-weight: bold; }\n"
"QGraphicsView { background-color: #000; border: 1px solid #59606a; }\n"
"QLabel#notes { background-color: #191b1e; border: 1px solid #454a52; padding: 6px; }\n"
"QDoubleSpinBox, QComboBox { background-color: #353941; border: 1px solid #69717d; padding: 4px; }")
        self.root_layout = QVBoxLayout(self.centralwidget)
        self.root_layout.setSpacing(8)
        self.root_layout.setObjectName(u"root_layout")
        self.main_splitter = QSplitter(self.centralwidget)
        self.main_splitter.setObjectName(u"main_splitter")
        self.main_splitter.setOrientation(Qt.Horizontal)
        self.observation_panel = QWidget(self.main_splitter)
        self.observation_panel.setObjectName(u"observation_panel")
        self.observation_layout = QVBoxLayout(self.observation_panel)
        self.observation_layout.setObjectName(u"observation_layout")
        self.observation_layout.setContentsMargins(0, 0, 0, 0)
        self.us_title = QLabel(self.observation_panel)
        self.us_title.setObjectName(u"us_title")
        self.us_title.setStyleSheet(u"font-size: 14pt; font-weight: bold;")

        self.observation_layout.addWidget(self.us_title)

        self.us_view = QGraphicsView(self.observation_panel)
        self.us_view.setObjectName(u"us_view")
        self.us_view.setMinimumSize(QSize(640, 480))

        self.observation_layout.addWidget(self.us_view)

        self.us_status = QLabel(self.observation_panel)
        self.us_status.setObjectName(u"us_status")

        self.observation_layout.addWidget(self.us_status)

        self.robot_state_group = QGroupBox(self.observation_panel)
        self.robot_state_group.setObjectName(u"robot_state_group")
        self.robot_state_layout = QVBoxLayout(self.robot_state_group)
        self.robot_state_layout.setObjectName(u"robot_state_layout")
        self.robot_status = QLabel(self.robot_state_group)
        self.robot_status.setObjectName(u"robot_status")
        self.robot_status.setStyleSheet(u"font-weight: bold; color: #ff6666;")

        self.robot_state_layout.addWidget(self.robot_status)

        self.robot_pose = QLabel(self.robot_state_group)
        self.robot_pose.setObjectName(u"robot_pose")
        font = QFont()
        font.setFamilies([u"Monospace"])
        font.setPointSize(12)
        self.robot_pose.setFont(font)

        self.robot_state_layout.addWidget(self.robot_pose)

        self.joint_state = QLabel(self.robot_state_group)
        self.joint_state.setObjectName(u"joint_state")
        self.joint_state.setFont(font)
        self.joint_state.setWordWrap(True)

        self.robot_state_layout.addWidget(self.joint_state)

        self.wrench_state = QLabel(self.robot_state_group)
        self.wrench_state.setObjectName(u"wrench_state")
        self.wrench_state.setFont(font)

        self.robot_state_layout.addWidget(self.wrench_state)

        self.inference_status = QLabel(self.robot_state_group)
        self.inference_status.setObjectName(u"inference_status")

        self.robot_state_layout.addWidget(self.inference_status)


        self.observation_layout.addWidget(self.robot_state_group)

        self.main_splitter.addWidget(self.observation_panel)
        self.planning_panel = QWidget(self.main_splitter)
        self.planning_panel.setObjectName(u"planning_panel")
        self.planning_layout = QVBoxLayout(self.planning_panel)
        self.planning_layout.setObjectName(u"planning_layout")
        self.planning_layout.setContentsMargins(0, 0, 0, 0)
        self.cbct_group = QGroupBox(self.planning_panel)
        self.cbct_group.setObjectName(u"cbct_group")
        self.cbct_layout = QVBoxLayout(self.cbct_group)
        self.cbct_layout.setObjectName(u"cbct_layout")
        self.vtk_volume = QVBoxLayout()
        self.vtk_volume.setObjectName(u"vtk_volume")

        self.cbct_layout.addLayout(self.vtk_volume)

        self.cbct_path = QLabel(self.cbct_group)
        self.cbct_path.setObjectName(u"cbct_path")
        self.cbct_path.setWordWrap(True)

        self.cbct_layout.addWidget(self.cbct_path)

        self.load_cbct = QPushButton(self.cbct_group)
        self.load_cbct.setObjectName(u"load_cbct")

        self.cbct_layout.addWidget(self.load_cbct)


        self.planning_layout.addWidget(self.cbct_group)

        self.atlas_group = QGroupBox(self.planning_panel)
        self.atlas_group.setObjectName(u"atlas_group")
        self.atlas_layout = QVBoxLayout(self.atlas_group)
        self.atlas_layout.setObjectName(u"atlas_layout")
        self.atlas_status = QLabel(self.atlas_group)
        self.atlas_status.setObjectName(u"atlas_status")
        self.atlas_status.setStyleSheet(u"color: #ff6666; font-weight: bold;")

        self.atlas_layout.addWidget(self.atlas_status)

        self.atlas_coordinate = QLabel(self.atlas_group)
        self.atlas_coordinate.setObjectName(u"atlas_coordinate")
        self.atlas_coordinate.setFont(font)

        self.atlas_layout.addWidget(self.atlas_coordinate)

        self.atlas_offset = QLabel(self.atlas_group)
        self.atlas_offset.setObjectName(u"atlas_offset")
        self.atlas_offset.setFont(font)

        self.atlas_layout.addWidget(self.atlas_offset)

        self.atlas_goal = QLabel(self.atlas_group)
        self.atlas_goal.setObjectName(u"atlas_goal")
        self.atlas_goal.setWordWrap(True)

        self.atlas_layout.addWidget(self.atlas_goal)

        self.atlas_belief = QLabel(self.atlas_group)
        self.atlas_belief.setObjectName(u"atlas_belief")
        self.atlas_belief.setWordWrap(True)

        self.atlas_layout.addWidget(self.atlas_belief)

        self.atlas_model = QLabel(self.atlas_group)
        self.atlas_model.setObjectName(u"atlas_model")
        self.atlas_model.setStyleSheet(u"color: #aeb7c4; font-size: 10pt;")
        self.atlas_model.setWordWrap(True)

        self.atlas_layout.addWidget(self.atlas_model)

        self.atlas_contract = QLabel(self.atlas_group)
        self.atlas_contract.setObjectName(u"atlas_contract")
        self.atlas_contract.setStyleSheet(u"color: #d6b66b; font-size: 10pt;")
        self.atlas_contract.setWordWrap(True)

        self.atlas_layout.addWidget(self.atlas_contract)


        self.planning_layout.addWidget(self.atlas_group)

        self.jog_group = QGroupBox(self.planning_panel)
        self.jog_group.setObjectName(u"jog_group")
        self.jog_outer_layout = QVBoxLayout(self.jog_group)
        self.jog_outer_layout.setObjectName(u"jog_outer_layout")
        self.jog_options_layout = QHBoxLayout()
        self.jog_options_layout.setObjectName(u"jog_options_layout")
        self.jog_step_label = QLabel(self.jog_group)
        self.jog_step_label.setObjectName(u"jog_step_label")

        self.jog_options_layout.addWidget(self.jog_step_label)

        self.jog_step_mm = QDoubleSpinBox(self.jog_group)
        self.jog_step_mm.setObjectName(u"jog_step_mm")
        self.jog_step_mm.setMinimum(0.100000000000000)
        self.jog_step_mm.setMaximum(50.000000000000000)
        self.jog_step_mm.setSingleStep(0.100000000000000)
        self.jog_step_mm.setValue(1.000000000000000)

        self.jog_options_layout.addWidget(self.jog_step_mm)

        self.jog_frame_label = QLabel(self.jog_group)
        self.jog_frame_label.setObjectName(u"jog_frame_label")

        self.jog_options_layout.addWidget(self.jog_frame_label)

        self.jog_frame = QComboBox(self.jog_group)
        self.jog_frame.setObjectName(u"jog_frame")

        self.jog_options_layout.addWidget(self.jog_frame)


        self.jog_outer_layout.addLayout(self.jog_options_layout)

        self.jog_buttons_layout = QGridLayout()
        self.jog_buttons_layout.setObjectName(u"jog_buttons_layout")
        self.jog_x_neg = QPushButton(self.jog_group)
        self.jog_x_neg.setObjectName(u"jog_x_neg")

        self.jog_buttons_layout.addWidget(self.jog_x_neg, 0, 0, 1, 1)

        self.jog_x_pos = QPushButton(self.jog_group)
        self.jog_x_pos.setObjectName(u"jog_x_pos")

        self.jog_buttons_layout.addWidget(self.jog_x_pos, 0, 1, 1, 1)

        self.jog_y_neg = QPushButton(self.jog_group)
        self.jog_y_neg.setObjectName(u"jog_y_neg")

        self.jog_buttons_layout.addWidget(self.jog_y_neg, 1, 0, 1, 1)

        self.jog_y_pos = QPushButton(self.jog_group)
        self.jog_y_pos.setObjectName(u"jog_y_pos")

        self.jog_buttons_layout.addWidget(self.jog_y_pos, 1, 1, 1, 1)

        self.jog_z_neg = QPushButton(self.jog_group)
        self.jog_z_neg.setObjectName(u"jog_z_neg")

        self.jog_buttons_layout.addWidget(self.jog_z_neg, 2, 0, 1, 1)

        self.jog_z_pos = QPushButton(self.jog_group)
        self.jog_z_pos.setObjectName(u"jog_z_pos")

        self.jog_buttons_layout.addWidget(self.jog_z_pos, 2, 1, 1, 1)


        self.jog_outer_layout.addLayout(self.jog_buttons_layout)

        self.enable_jog = QCheckBox(self.jog_group)
        self.enable_jog.setObjectName(u"enable_jog")

        self.jog_outer_layout.addWidget(self.enable_jog)

        self.jog_contract = QLabel(self.jog_group)
        self.jog_contract.setObjectName(u"jog_contract")
        self.jog_contract.setStyleSheet(u"color: #d6b66b;")
        self.jog_contract.setWordWrap(True)

        self.jog_outer_layout.addWidget(self.jog_contract)


        self.planning_layout.addWidget(self.jog_group)

        self.main_splitter.addWidget(self.planning_panel)

        self.root_layout.addWidget(self.main_splitter)

        self.notes = QLabel(self.centralwidget)
        self.notes.setObjectName(u"notes")
        self.notes.setMinimumSize(QSize(0, 42))

        self.root_layout.addWidget(self.notes)

        self.bottom_bar = QHBoxLayout()
        self.bottom_bar.setObjectName(u"bottom_bar")
        self.record = QPushButton(self.centralwidget)
        self.record.setObjectName(u"record")

        self.bottom_bar.addWidget(self.record)

        self.stop_record = QPushButton(self.centralwidget)
        self.stop_record.setObjectName(u"stop_record")
        self.stop_record.setEnabled(False)

        self.bottom_bar.addWidget(self.stop_record)

        self.start_inference = QPushButton(self.centralwidget)
        self.start_inference.setObjectName(u"start_inference")

        self.bottom_bar.addWidget(self.start_inference)

        self.stop_inference = QPushButton(self.centralwidget)
        self.stop_inference.setObjectName(u"stop_inference")
        self.stop_inference.setEnabled(False)

        self.bottom_bar.addWidget(self.stop_inference)

        self.bottom_spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.bottom_bar.addItem(self.bottom_spacer)

        self.reset_robot = QPushButton(self.centralwidget)
        self.reset_robot.setObjectName(u"reset_robot")

        self.bottom_bar.addWidget(self.reset_robot)

        self.stop_robot = QPushButton(self.centralwidget)
        self.stop_robot.setObjectName(u"stop_robot")

        self.bottom_bar.addWidget(self.stop_robot)

        self.quit = QPushButton(self.centralwidget)
        self.quit.setObjectName(u"quit")

        self.bottom_bar.addWidget(self.quit)


        self.root_layout.addLayout(self.bottom_bar)

        DeepUSNavConsole.setCentralWidget(self.centralwidget)

        self.retranslateUi(DeepUSNavConsole)

        QMetaObject.connectSlotsByName(DeepUSNavConsole)
    # setupUi

    def retranslateUi(self, DeepUSNavConsole):
        DeepUSNavConsole.setWindowTitle(QCoreApplication.translate("DeepUSNavConsole", u"DeepUSNav \u2014 FR3 Operator Console", None))
        self.us_title.setText(QCoreApplication.translate("DeepUSNavConsole", u"Live ultrasound", None))
        self.us_status.setText(QCoreApplication.translate("DeepUSNavConsole", u"US: waiting", None))
        self.robot_state_group.setTitle(QCoreApplication.translate("DeepUSNavConsole", u"FR3 state", None))
        self.robot_status.setText(QCoreApplication.translate("DeepUSNavConsole", u"Robot: disconnected", None))
        self.robot_pose.setText(QCoreApplication.translate("DeepUSNavConsole", u"waiting for robot pose", None))
        self.joint_state.setText(QCoreApplication.translate("DeepUSNavConsole", u"q: waiting", None))
        self.wrench_state.setText(QCoreApplication.translate("DeepUSNavConsole", u"External wrench: waiting", None))
        self.inference_status.setText(QCoreApplication.translate("DeepUSNavConsole", u"Inference: not connected", None))
        self.cbct_group.setTitle(QCoreApplication.translate("DeepUSNavConsole", u"CBCT (visualization only)", None))
        self.cbct_path.setText(QCoreApplication.translate("DeepUSNavConsole", u"No CBCT loaded", None))
        self.load_cbct.setText(QCoreApplication.translate("DeepUSNavConsole", u"Load CBCT DICOM directory", None))
        self.atlas_group.setTitle(QCoreApplication.translate("DeepUSNavConsole", u"Population Atlas \u2014 L4 localisation", None))
        self.atlas_status.setText(QCoreApplication.translate("DeepUSNavConsole", u"Atlas: waiting for inference node", None))
        self.atlas_coordinate.setText(QCoreApplication.translate("DeepUSNavConsole", u"u-hat: --", None))
        self.atlas_offset.setText(QCoreApplication.translate("DeepUSNavConsole", u"offset to L4 [mm]: --", None))
        self.atlas_goal.setText(QCoreApplication.translate("DeepUSNavConsole", u"L4 goal cost: --", None))
        self.atlas_belief.setText(QCoreApplication.translate("DeepUSNavConsole", u"retrieval belief: --", None))
        self.atlas_model.setText(QCoreApplication.translate("DeepUSNavConsole", u"model: --", None))
        self.atlas_contract.setText(QCoreApplication.translate("DeepUSNavConsole", u"Atlas coordinates are anatomical estimates, not robot Cartesian coordinates. Do not send them directly to the controller.", None))
        self.jog_group.setTitle(QCoreApplication.translate("DeepUSNavConsole", u"Manual Cartesian jog", None))
        self.jog_step_label.setText(QCoreApplication.translate("DeepUSNavConsole", u"Step [mm]", None))
        self.jog_frame_label.setText(QCoreApplication.translate("DeepUSNavConsole", u"Frame", None))
        self.jog_x_neg.setText(QCoreApplication.translate("DeepUSNavConsole", u"\u2212X", None))
        self.jog_x_pos.setText(QCoreApplication.translate("DeepUSNavConsole", u"+X", None))
        self.jog_y_neg.setText(QCoreApplication.translate("DeepUSNavConsole", u"\u2212Y", None))
        self.jog_y_pos.setText(QCoreApplication.translate("DeepUSNavConsole", u"+Y", None))
        self.jog_z_neg.setText(QCoreApplication.translate("DeepUSNavConsole", u"\u2212Z", None))
        self.jog_z_pos.setText(QCoreApplication.translate("DeepUSNavConsole", u"+Z", None))
        self.enable_jog.setText(QCoreApplication.translate("DeepUSNavConsole", u"Enable manual jog (operator acknowledgement)", None))
        self.jog_contract.setText(QCoreApplication.translate("DeepUSNavConsole", u"Each click sends one relative displacement request. The real-time controller must validate, interpolate and execute it.", None))
        self.notes.setText(QCoreApplication.translate("DeepUSNavConsole", u"Ready. Commands remain locked until manual jog is enabled.", None))
        self.record.setText(QCoreApplication.translate("DeepUSNavConsole", u"Start rosbag2", None))
        self.stop_record.setText(QCoreApplication.translate("DeepUSNavConsole", u"Stop recording", None))
        self.start_inference.setText(QCoreApplication.translate("DeepUSNavConsole", u"Start inference", None))
        self.stop_inference.setText(QCoreApplication.translate("DeepUSNavConsole", u"Stop inference", None))
        self.reset_robot.setText(QCoreApplication.translate("DeepUSNavConsole", u"Reset controller", None))
        self.stop_robot.setText(QCoreApplication.translate("DeepUSNavConsole", u"STOP MOTION", None))
        self.quit.setText(QCoreApplication.translate("DeepUSNavConsole", u"Quit", None))
    # retranslateUi

