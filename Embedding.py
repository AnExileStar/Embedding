import os
import jieba
import chardet
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
from typing import List, Dict, Tuple

# 配置路径
class Config:
    BASE_DIR = "D:/project/data/jyxstxtqj/"
    NOVEL_FILES = [
        "三十三剑客图.txt", "书剑恩仇录.txt", "侠客行.txt", "倚天屠龙记.txt",
        "天龙八部.txt", "射雕英雄传.txt", "白马啸西风.txt", "碧血剑.txt",
        "神雕侠侣.txt", "笑傲江湖.txt", "越女剑.txt", "连城诀.txt",
        "雪山飞狐.txt", "飞狐外传.txt", "鸳鸯刀.txt", "鹿鼎记.txt"
    ]
    NAME_FILE = os.path.join(BASE_DIR, "name.txt")
    STOPWORDS_FILE = "D:/project/data/stopwords.txt"
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

class NovelProcessor:
    """处理金庸小说文本的类"""
    
    def __init__(self):
        self.novels = []
        self.name_dict = []
        self.stopwords = set()
        
    def load_names(self) -> None:
        """加载人物名列表"""
        with open(Config.NAME_FILE, "r", encoding="utf-8") as f:
            self.name_dict = f.read().split(" ")
        jieba.load_userdict(self.name_dict)
        print(f"加载了 {len(self.name_dict)} 个人名")
        
    def load_stopwords(self) -> None:
        """加载停用词表"""
        with open(Config.STOPWORDS_FILE, 'r', encoding='utf-8') as f:
            self.stopwords = {line.strip() for line in f}
        print(f"加载了 {len(self.stopwords)} 个停用词")
        
    def detect_encoding(self, file_path: str) -> str:
        """检测文件编码"""
        with open(file_path, 'rb') as f:
            return chardet.detect(f.read())["encoding"]
        
    def clean_text(self, text: str) -> str:
        """清洗文本"""
        # 去除标点、数字、空白等
        text = text.replace("\u3000", "").replace(" ", "").replace("　", "")
        text = "".join(char for char in text if char not in "，。？！；：‘'""（）《》【】")
        text = "".join(char for char in text if not char.isdigit())
        return text
        
    def load_novels(self) -> None:
        """加载所有小说文本"""
        for novel_file in Config.NOVEL_FILES:
            file_path = os.path.join(Config.BASE_DIR, novel_file)
            encoding = self.detect_encoding(file_path)
            
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                cleaned_content = self.clean_text(content)
                self.novels.append(cleaned_content)
                
        print(f"成功加载 {len(self.novels)} 部小说")
        
    def tokenize_novels(self) -> List[List[str]]:
        """分词处理"""
        tokenized = []
        for novel in self.novels:
            words = jieba.lcut(novel)
            # 去除停用词
            words = [word for word in words if word not in self.stopwords]
            tokenized.append(words)
            
        # 保存分词结果
        output_path = os.path.join(Config.OUTPUT_DIR, "tokenized.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            for words in tokenized:
                f.write(" ".join(words) + " ")
                
        return tokenized

class EmbeddingAnalyzer:
    """处理嵌入向量分析的类"""
    
    def __init__(self, name_dict: List[str]):
        self.name_dict = name_dict
        self.model = None
        self.valid_names = []
        self.embeddings = []
        
    def train_word2vec(self, tokenized_novels: List[List[str]]) -> None:
        """训练Word2Vec模型"""
        print("开始训练Word2Vec模型...")
        self.model = Word2Vec(
            tokenized_novels,
            vector_size=150,  # 增加向量维度以捕捉更多特征
            window=8,        # 增大上下文窗口
            min_count=2,      # 忽略低频词
            workers=6,       # 使用更多线程
            sg=1,            # 使用skip-gram算法
            hs=1,            # 使用hierarchical softmax
            epochs=15        # 增加训练轮次
        )
        
        model_path = os.path.join(Config.OUTPUT_DIR, "word2vec.model")
        self.model.save(model_path)
        print(f"模型已保存到 {model_path}")
        
    def load_model(self) -> None:
        """加载预训练模型"""
        model_path = os.path.join(Config.OUTPUT_DIR, "word2vec.model")
        self.model = Word2Vec.load(model_path)
        
    def extract_name_embeddings(self) -> None:
        """提取人物名嵌入向量"""
        for name in self.name_dict:
            if name in self.model.wv:
                self.embeddings.append(self.model.wv[name])
                self.valid_names.append(name)
                
        print(f"成功提取 {len(self.valid_names)}/{len(self.name_dict)} 个人物名的嵌入向量")
        
    def perform_pca(self) -> Tuple[np.ndarray, np.ndarray]:
        """执行PCA降维"""
        X = np.array(self.embeddings)
        
        # 2D PCA
        pca_2d = PCA(n_components=2, random_state=42)
        X_2d = pca_2d.fit_transform(X)
        print(f"2D PCA解释方差比例: {pca_2d.explained_variance_ratio_}")
        
        # 3D PCA
        pca_3d = PCA(n_components=3, random_state=42)
        X_3d = pca_3d.fit_transform(X)
        print(f"3D PCA解释方差比例: {pca_3d.explained_variance_ratio_}")
        
        return X_2d, X_3d
    
    def visualize(self, X_2d: np.ndarray, X_3d: np.ndarray) -> None:
        """可视化结果"""
        self._plot_2d(X_2d)
        self._plot_3d(X_3d)
        
    def _plot_2d(self, X_2d: np.ndarray) -> None:
        """绘制2D散点图"""
        plt.figure(figsize=(15, 15))
        plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6, color='blue')
        
        # 只标注主要角色名，避免重叠
        main_characters = ["郭靖", "黄蓉", "杨过", "小龙女", "张无忌", 
                         "令狐冲", "乔峰", "段誉", "虚竹", "韦小宝"]
        
        for i, name in enumerate(self.valid_names):
            if name in main_characters:
                plt.annotate(name, (X_2d[i, 0], X_2d[i, 1]), 
                            fontsize=12, 
                            arrowprops=dict(arrowstyle="->", color="red"))
                
        plt.title("金庸小说人物名嵌入向量 - 2D PCA", fontsize=16)
        plt.xlabel("主成分1", fontsize=14)
        plt.ylabel("主成分2", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(Config.OUTPUT_DIR, "2d_visualization.png")
        plt.savefig(output_path, dpi=300)
        plt.show()
        
    def _plot_3d(self, X_3d: np.ndarray) -> None:
        """绘制3D散点图"""
        fig = px.scatter_3d(
            x=X_3d[:, 0], y=X_3d[:, 1], z=X_3d[:, 2],
            text=self.valid_names,
            title="金庸小说人物名嵌入向量 - 3D PCA",
            opacity=0.7,
            size_max=10,
            labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
            hover_name=self.valid_names
        )
        
        # 调整布局
        fig.update_layout(
            scene=dict(
                xaxis_title='主成分1',
                yaxis_title='主成分2',
                zaxis_title='主成分3'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            hoverlabel=dict(font_size=14)
        )
        
        # 只显示主要角色的标签
        main_char_indices = [i for i, name in enumerate(self.valid_names) 
                            if name in ["郭靖", "黄蓉", "杨过", "小龙女", "张无忌"]]
        for i in main_char_indices:
            fig.update_traces(
                textposition='top center',
                selector=dict(text=self.valid_names[i])
            )
            
        output_path = os.path.join(Config.OUTPUT_DIR, "3d_visualization.html")
        fig.write_html(output_path)
        fig.show()

def main():
    # 1. 数据预处理
    processor = NovelProcessor()
    processor.load_names()
    processor.load_stopwords()
    processor.load_novels()
    tokenized_novels = processor.tokenize_novels()
    
    # 2. 训练和提取嵌入向量
    analyzer = EmbeddingAnalyzer(processor.name_dict)
    analyzer.train_word2vec(tokenized_novels)
    analyzer.extract_name_embeddings()
    
    # 3. 降维和可视化
    X_2d, X_3d = analyzer.perform_pca()
    analyzer.visualize(X_2d, X_3d)

if __name__ == "__main__":
    main()
