import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QTextEdit, QFileDialog, QLabel, QHBoxLayout, QComboBox
)
from PyQt5.QtCore import Qt
from movie_preference_analysis import MoviePreferenceAnalyzer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("电影偏好分析系统")
        self.setGeometry(200, 200, 900, 600)

        self.analyzer = MoviePreferenceAnalyzer()
        self.user_ids = []

        # 主界面
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 顶部操作区
        top_layout = QHBoxLayout()
        self.info_label = QLabel("请选择CSV文件进行分析")
        top_layout.addWidget(self.info_label)

        self.load_button = QPushButton("加载CSV文件并分析")
        self.load_button.clicked.connect(self.load_and_analyze)
        top_layout.addWidget(self.load_button)

        layout.addLayout(top_layout)

        # 推荐区
        recommend_layout = QHBoxLayout()
        self.user_combo = QComboBox()
        self.user_combo.setEnabled(False)
        recommend_layout.addWidget(QLabel("选择用户:"))
        recommend_layout.addWidget(self.user_combo)

        self.recommend_button = QPushButton("为该用户推荐电影")
        self.recommend_button.clicked.connect(self.recommend_movies)
        self.recommend_button.setEnabled(False)
        recommend_layout.addWidget(self.recommend_button)

        layout.addLayout(recommend_layout)

        # 输出区
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        layout.addWidget(self.text_area)

    def load_and_analyze(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择CSV文件", "", "CSV Files (*.csv)")
        if file_path:
            self.info_label.setText(f"已选择文件: {file_path}")
            self.text_area.append("正在分析数据，请稍候...\n")
            QApplication.processEvents()
            self.analyzer.run_full_analysis(file_path)
            self.text_area.append("分析完成！\n")
            # 填充用户下拉框
            self.user_ids = list(self.analyzer.user_features['user_id'])
            self.user_combo.clear()
            self.user_combo.addItems([str(uid) for uid in self.user_ids])
            self.user_combo.setEnabled(True)
            self.recommend_button.setEnabled(True)

    def recommend_movies(self):
        if not self.user_ids:
            self.text_area.append("请先加载数据！")
            return
        user_id = int(self.user_combo.currentText())
        self.text_area.append(f"\n为用户 {user_id} 推荐电影：\n")
        recs = self.analyzer.get_recommendations(user_id, 5, 'user_based')
        if not recs:
            self.text_area.append("无推荐结果。\n")
        else:
            for i, movie in enumerate(recs, 1):
                self.text_area.append(f"{i}. {movie['电影名称']} ({movie['类型']}) - 预测评分: {movie['预测评分']}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())