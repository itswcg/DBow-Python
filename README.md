#### DBow_Python
DBow_Python是[DBow](https://github.com/itswcg/DBow)的python精简版。  


* ORB特征提取
* Kmeans聚类
* 字典树结构保存字典文件 
 
生成的字典文件能直接用于orb slam2中。

#### 使用
* Python2 + Opencv3 + Numpy
##### 生成特定场景的视觉词典
* 替换imgages中的图片，图片尽可能多
* 修改main.py中N，K，L，一般K设为10，L设为5
* 在orb.py中，修改每幅图像提取特征点的数量，为了增加字典中单词的数量

##### 比较图像之间的相似度
创建视觉词典就是为了比较两幅图像之间的相似度，只是采用bow模型，降低复杂度而已。每幅图像可以用视觉词典中的单词向量表示，再计算向量之间的余弦相似度。在字典树中，叶子节点就是单词，比如创建5，3的树，就会有`K**L`个单词，一幅图像就可以表示为125维向量。  
结果如下：  
![](https://res.cloudinary.com/itswcg/image/upload/v1523539603/dbow_jznss6.png)
