舆情监控、话题聚类分析、短文本聚类、话题推荐、问答系统

##  基于大数据的互联网热点话题挖掘的研究与实现

随着web2.0到来，新闻门户、微博和知乎等信息发布平台取代了传统媒体。海量短文本和图片等非结构化数据让手动挖掘并追踪热点变得异常困难。需要研究自动采集短文本数据，并自动进行话题挖掘、追踪的平台。
因此,基于海量数据的背景,设计并实现能**实时检测**并**自动发现**和**跟踪**各种互联网热点话题的话题检测与追踪方案具有重要的意义。
最后把获取的数据以及大量非结构化数据持久化到大数据存储平台中。
针对以上分析,并结合互联网热点话题的特点与大数据平台,设计该平台可以从以下几个方面入手：

第一,设计并完成针对新闻门户网站、社交媒体等平台新发布信息的**实时增量爬虫**,并现有的爬虫框架Heritrix 、Heritri、JSpider进行分布式拓展。爬取网页标题和正文报道，以及社交媒体的话题和评论。将数据存储到HBase后，爬虫系统开始睡眠，等待下次任务开始。将采集到的数据存储到HBase上，然后进行话题挖掘，构建**知识图谱**。

第二,使用流行的分词工具为采集得到的文本进行预处理，提出基于报道、话题上关键特征与命名实体的文本表示模型,并结合基于**朴素贝叶斯**分类划分的**三层聚类算法**，可以在给定的话题中对所有的文本进行聚类，找出该话题下的子话题，然后可以对相应子话题进行跟踪和预测。

思路:先使用分类算法把话题按照主流新闻门户的分类（财经、体育、生活等）方式进行分类，部分特征和标签（关键词、时间信息、文本来源和作者等）在爬虫时获取。然后在每个子话题进行聚类，可以提高聚类的效率。
传统的长文本话题推荐方法：

- 无监督学习方法的话题发现:基于神经网络的 SOM 和 FCM
- 基于有监督学习方法的话题发现：KNN 、决策树、朴素贝叶斯

短文本的聚类算法：

1. 增量聚类：spark并行化的Single-Pass算法，
2. 基于划分的聚类：自定义文本距离公式，计算文本之间的相似度。然后利用k-means算法对评论、弹幕和标题等短文本进行聚类。质心词汇代表了主流的观点。
3. 层次聚类：过于精细，不适合处理大规模数据，可以对增量数据按照层次结构做展示。
4. 密度聚类：DBSCN

第三，设计出能提供针对性意见的**智能决策支持系统**，根据语料回答问题，帮助政府理解社会问题的产生、应对策略等。



第四,实现针对海量互联网资讯的实时**热点话题的可视化平台**,通过平台展示最新发现的热点话题资讯,并利用可视化工具对话题统计信息进行可视化展示。本文所提出的方法与设计的系统,能够在海量数据的前提下进行新闻和博客数据的高效采集,对采集的文本分别进行数据筛选和热点话题的发现与跟踪,并最终展示在前台网页中,具有较高的实用价值,并已接受有效性的检验。

1. 热点话题的发现与跟踪：自动检测热点话题，对热点话题进行跟踪和情感识别，预测话题未来走向，对于突发事件和敏感信息进行预警。重大事件发生时，微博平台会迅速聚集大量转发与讨论量，舆情监控系统可以迅速检测到相关事件，了解事件发生的前因后果及最新动态。政府部门在获取相关信息后，可及时采取措施解决问题，化解矛盾，积极引导群众的正向口碑效应，稳定民心，掌控局势。
2. 话题溯源：跟踪微博话题发布者以及转发者，有助于理解新闻传播过程中发生的变化。核心账号和博文是事件传播的推动者和源头，利用这些账号便于政府推广、科普以及辟谣。
3. 不同地域和不同年龄群体参与度热度图：可以看到不同群体对于热点事件的关注度。同类人对于事件通常有着相似的观点，找到主要群体，有利于政府理解人们对于事件的看法，并进行针对性的进行宣传。

第五,利用Hadoop大数据处理平台,结合Map-Reduce并行计算模型和HBase非结构化数据库,**批量处理和存储**海量的舆情资讯,挖掘每天热点话题;

ACL、EMNLP、NAACL、COLING、ICLR、AAAI、CoNLL、NLPCC

## 关键技术：基于深度学习的结构化数据问答方法研究

问答系统：使用结构化数据回答问题，主要方式是语义解析。自然语言转化为sql或者sparql。
传统语义解析：模板、基于语法或者句法、机器学习方法可以自动从问题-答案标注数据中学习转换规则。

针对知识图谱简单关系问答，提出联合生成、复制和改写模板的方法。





**[ACL](https://www.aclweb.org/portal)**、**[EMNLP](http://emnlp2018.org/)**、**[NAACL](http://naacl.org/)**  可以说是 NLP 领域的四大顶会

Kmeans文本聚类四步走

 步骤一、对文本进行切词和去除停用词。（jieba）
**jieba分词对文本进行预处理，同时利用网上下的停用词文档结合正则表达式去除语气词和数字等，去除后的效果如下图所示：**

 步骤二、计算文本特征并构建 VSM（向量空间模型）。

步骤三、使用 K-means 算法进行聚类。

 步骤四、对新文档进行分类并计算分类成功率





爬虫、文档集持久化、话题挖掘；三级话题挖掘；文本聚类；

项目需求

```
指定话题/日期==>话题历史热度（类似百度搜索指数）、话题子类、地区和门户网站的话题热度区别
fakeNews识别
```



需求：ccf A类论文集合。最小单位是论文名字+引用+论文下载地址

问题：词袋无法识别新的术语，在分词是会错误地把新术语分割；



热门话题跟踪以及热点事件的检测和发现

文本聚类：Text clustering

关联话题：Apriori算法；输入：

TDT：Hot Topic Detection and Tracking

知识图谱：

```
“实体-关系-实体”
或“实体-属性-属性值”三元组构成，大量这样的三元组交织连接
形成了一个在物理层面和逻辑层面上同时存在的知识网络。
开放领域和垂直领域
```

Single-Pass 算法

```

```



LDA 模型：

```
广泛使用的主题模型：隐含狄利克雷分布Latent Dirichlet Allocation
```

Stochastic block model：随机块模型

```

```

Word2vec:

```
是一群用来产生[词向量]的神经网络模型
```



(TF-PDF）TF-Proportional Document Frequency

```

```

主题模型

```
LDA (Blei et al., 2003) and GSDMM (Yin and Wang, 2014),

[1]Blei, David M , Andrew Y , et al. Latent Dirichlet Allocation.[J]. Journal of Machine Learning Research, 2003.

Yin J , Wang J . A dirichlet multinomial mixture model-based approach for short text clustering[M]. ACM, 2014.
主题模型提高BERT在语意相似度检测的性能
```

重要性：

1. 倾听民生重要通道
2. 实时获取反馈
3. 提出准确建议
4. 比传统新闻媒体更早出现、传播更快
5. 数据量大、多数维短文本而且表达不规范，含有大量新词汇。

一个事件event、topic是由多个关键词代表。

从海量数据中高效提取信息

问题：

1. 没有考虑时间对事件的影响，静态模型。话题能量模型
2. 这项任务的主要困难是：很难处理短、不规则、嘈杂的社交媒体文本。很难生成随着时间而发展的事件的表示。由于有大量的推文，它是计算密集型的。

文本向量化算法：

1. 采用基于降维的词典模型，提取关键的特征，将高维度的稀疏文本向量降低维度后进行处理。例如TF-IDF

   ```
   Ramos J. Using tf-idf to determine word relevance in document queries[C]//Proceedings of the first instructional conference on machine learning. 2003, 242(1): 29-48.
   ```

   

2. 基于深度学习的向量化方法，包括
   word2Vec:

   word2vec是一个基于深度学习的开源学习工具，可以将词转化为词向量。词向量表示的形式充分考虑了文本中词之间的语义关系，扩展了语义相似关系选择的特征集，使得特征集更加完整，提高了文本分类的性能。有两个重要的模型：CBOW模型和Skip-gram模型

   ```
   [1] Mikolov T ,  Chen K ,  Corrado G , et al. Efficient Estimation of Word Representations in Vector Space[J]. Computer Science, 2013.
   ```

   

   . 

文本聚类算法：

1. 基于相似性的聚类算法的
   single-Pass

   DBSACN

   K-means

2. 基于概率模型的聚类算法在处理高维文本数据方面表现出了更好的性能

3. 狄利克雷多项混合(DMM)模型，在(Yin和Wang2014)中

主题模型（Topic Model）抽取算法：

潜在的狄利克雷分配(LDA)(Blei，Ng，和Jordan2003)，是一个定向的无监督的主题模型

1. 陈燕、Hadi使用single-pass聚类算法和二分类的SVM分类器发现新话题，但是没有考虑到时间对话题的影响，属于静态模型。而且简单的SVM分类器需要事前选取合适的特征、人为标注标签预训练，这在处理大规模数据集时会很困难。

```
Chen Y ,  Amiri H ,  Li Z , et al. Emerging topic detection for organizations from microblogs[C]// International Acm Sigir Conference on Research & Development in Information Retrieval. ACM, 2013:43.
```

2.  Maksim、Hady提出了引入了半监督的CompareLDA算法用来鉴别指定主题下的文档。目的是研究CompareLDA是否可以从同一语料库中获得不同的主题，但有主题之间必须是对立的，而且需要预先标记样本训练模型。

```
Tkachenko M, Lauw H W. Comparelda: A topic model for document comparison[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33(01): 7112-7119.
```

3. 邱兴发、邹桥沙使用BERT对语料集进行预训练，获得训练过的编码器，将文档和话题映射到同样的低维语义空间，然后计算他们之间的相似度。最后提出了能动态更替关键词的single-pass算法在公共的Twitter数据集上取得优越的聚类效果。

   

```
Qiu X, Zou Q, Richard Shi C J. Single-Pass On-Line Event Detection in Twitter Streams[C]//2021 13th International Conference on Machine Learning and Computing. 2021: 522-529.
```

TDT  task  BERT模型

```
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT:
pre-training of deep bidirectional transformers for language understanding. In
Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, (NAACLHLT 2019), Minneapolis, MN, USA, 4171-4186. https://doi.org/10.18653/v1/n19-
1423
```



有很多监督方法来处理特定领域的TDT任务，

Sakaki等人提出来二分类支持向量机 (SVM)从推特数据集实时检测地震和预测地震中心。

```
Sakaki Takeshi, Okazaki Makoto, and Matsuo Yutaka. 2010. Earthquake shakes Twitter users: real-time event detection by social sensors. In Proceedings of the 19th International Conference on World Wide Web (WWW2010), Raleigh, North Carolina, USA, 851–860. https://doi.org/10.1145/1772690.1772777
```

Nguyen等人引入递归神经网络RNN来检测事件模型。

```
Van Quan Nguyen, Tien Nguyen Anh, and Hyung-Jeong Yang. 2019. Realtime event detection using recurrent neural network in socialsensors.International Journal of Distributed Sensor Networks. 15, 6. https://doi.org/10.1177/
1550147719856492
```

Wang等人提出了一种基于LSTM（递归神经网络RNN的一个变种）的事件检测和总结的联合模型。

```
Zhongqing Wang and Yue Zhang. 2017. A neural model for joint event detection
and summarization. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI 2017), Melbourne, Australia, 4158-4164.
https://doi.org/ 10.24963/ijcai.2017/581
```

还有一些研究者专注于研究无监督的事件检测方法，因为社交媒体数据不能包含所有预先定义的事件。其中聚类算法应用最为广泛。

Hansi[26]结合了基于预测的词嵌入和层次聚类的特征，在社交媒体中进行事件检测。卡梅拉等人。

```
Hansi Hettiarachchi, Adedoyin-Olowe, Mariam, Jagdev Bhogal, and Mohamed
Medhat Gaber. 2020. Embed2Detect: Temporally clustered embedded words for
event detection in social media. CoRR. https://arxiv.org/abs/2006.05908
```



Charu[14]使用TF-IDF（术语频率逆文档频率）向量作为文本表示，tweet的平均表示作为事件表示，并将它们输入到一个基于增量聚类算法的系统中。然而，基于TF-IDF的相似性在事件检测中是不够的，因为单词的时间顺序和文本的语义和语法特征都是不够的。

```
Aggarwal, Charu C, and Subbian, Karthik. 2012. Event detection in social streams.
In Proceedings of the Twelfth SIAM International Conference on Data Mining,
Anaheim, California, USA, 624-635. https://doi.org/10.1137/1.9781611972825.54

Giridhar Kumaran, and James Allan. 2004. Text classification and named entities for new event detection, In Proceedings of the 27th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval,
Sheffield, UK, 297-304. https://doi.org/10.1145/1008992.1009044
```



混合方法结合了监督学习和无监督学习。Jagan[17]训练一个分类器来区分与事件相关的推文和与事件无关的推文。它可以帮助减少为聚类提供的数据量，从而提高系统的效率。Becker尔等人[18]首先进行聚类，然后使用SVM对聚类结果中是否包含关于相关信息进行分类。

```
Jagan Sankaranarayanan, Hanan Samet, Benjamin E Teitler, Michael D Lieberman, and Jon Sperling. 2009. TwitterStand: news in tweets. In 17th ACM SIGSPATIAL International Symposium on Advances in Geographic Information Systems, (ACM-GIS 2009), Seattle, Washington, USA, 42–51. https://doi.org/10.1145/
1653771.1653781
```

```
Hila Becker, Mor Naaman, and Luis Gravano. 2011. Beyond Trending Topics: RealWorld Event Identification on Twitter . In Proceedings of the Fifth International
Conference on Weblogs and Social Media, Barcelona, Catalonia, Spain, https:
//doi.org/10.7916/D81V5NVX
```

​	由相似度模型和概率模型产生的主题都有共同的问题，产生的topic是一个词或者多个词的加权。一个词难以很好地概括整个文档集的核心思想。多个词按权重排列，虽然可以比较全面的提及了文档的高频词汇，但是难以被解释和理解。如果想要进一步理解，还是要查看该话题下的多个文档。

​	常规的语义理解往往建立在问句和文本句子的相似度计算，然而语义理解和知识的本质在于关联。相似度的计算模型忽略了这一点。为此，需要进步强化话题的表示，利用知识图谱将话题发现算法产生的话题归纳成一个可以被理解的知识。

​	**知识图谱的构建使用sql等结构化查询语言有利于快速检索，而纯文本匹配需要更长的时间。**

知识图谱的知识来源于专业人士标注和专业数据库的格式化抓取，保证了极高的准确率。

构建知识图谱（开源+实时导入、知识来源于哪里）、使用知识图谱。

**基于知识图谱的问答系统的本质是把每个文本问题映射为关于知识谱图的结构化查询。**

2012 年 5 月份，Google 花重金收购 Metaweb 公司，并向外界正式发布其知识图谱(knowledge graph)。自此，知识图谱正式走入公众视野。开放领域大规模知识图谱纷纷出现，包括 NELL [15]，Freebase [10]，Dbpedia [6]，Probase [103]等。

知识图谱：知识图谱本质上是一种语义网络。其结点代表实体（entity）或者概念（concept），边代表实体/概念之间的各种语义关系。语义的本质是关联的三元组，在自然语言处理、问答系统和推荐系统领域产生了广泛应用。

RDF知识图谱：

知识图嵌入的目的是在向量空间中嵌入实体和知识图的关系(Bordes等。2011年；2013年；Wang等人。2014b；2014a；

文本蕴含

```
Silva V S, Freitas A, Handschuh S. Exploring knowledge graphs in an interpretable composite approach for text entailment[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33(01): 7023-7030.
```

知识图谱嵌入是知识图谱领域一个新的研究热点，旨在利用词向量的平移不变性将知识图谱中实

体和关系嵌入到低维向量空间，进而完成知识表示。



### 社会治理为什么要话题发现

传统的社会治理方式也无法把握新时代社会发展的要求，无法满足不同社会主体的多样化的需求。大数据时代要求社会治理必须能及时地参考多元信息。





### 构建知识图谱

1. 实体识别技术
2. 关系抽取技术

智能决策支持平台



### PPT

1. 标题

2. 目录

3. 背景：社会治理

   研究现状: 现有的热点事件发现大多以词为粒度将词语向量化之后，利用监督或无监督的机器学习算法对词集进行分类或者聚类，提取出热点词当做事件。

4. 现有的文本向量化（忽略词序、只关注语义不考虑文本之间的关系）

5. 相近的概念是知识图谱的嵌入算法用来补全知识图谱

6. 热点事件发现算法（不能增量更新、聚类结果不稳定、不能体现话题的演化关系）

#### 研究内容和技术方案

1. **知识图谱嵌入的文本向量化算法：**word2vec、

2. 知识图谱构建：指定实体和关系，爬虫抽取相应字段，存入设计好的数据库，遍历抽取三元组

   1. ![img](${picture}/wps1.jpg)![img](${picture}/wps2.jpg)![img](${picture}/wps3.jpg)

3. **热点事件发现模型：**

   1. 先对全量的文本聚类，得到每个文本所在的一级事件
   2. 利用聚类结果作为标签训练分类器，
   3. 后续增量数据先文本分类，然后每个一级事件下进行聚类，发现新的子事件
   4. 新老事件热度随时间衰减，新事件替换老事件需要重新训练分类器，驱逐旧事件的样本

4. 双层聚类的好处：

   1. 分类器可以将文本集合分割，减少噪声对聚类效果的干扰，子事件的聚类可以并行；
   2. 两次聚类生成新老话题的结构，树状事件演变结构

5. 多级存储：

   应用层，使用存储中间件的形式进行。

   针对文本数据和文本向量，根据文本的时间、访问频率、文本中实体词的数量、所在事件的热度和相关文本的频率预测可能会出现的热点信息。

   缓存调度模块会异步记录下每次没有命中的文本向量的信息；在系统负载较低的时候，评估其是否为潜在的高热度事件

6. 已经取得的进展：

7. 预期成果

8. 感谢



![image-20211125092219824](${picture}/image-20211125092219824.png)



# 术语

## 数据增强：

在自然语言处理领域，被验证为有效的数据增强算法相对要少很多，下面我们介绍几种常见方法。

- **同义词词典**（Thesaurus）：Zhang Xiang等人提出了Character-level Convolutional Networks for Text Classification，通过实验，他们发现可以将单词替换为它的同义词进行数据增强，这种同义词替换的方法可以在很短的时间内生成大量的数据。
- **随机插入**（Randomly Insert）：随机选择一个单词，选择它的一个同义词，插入原句子中的随机位置，举一个例子：“我爱中国” —> “喜欢我爱中国”。
- **随机交换**（Randomly Swap）：随机选择一对单词，交换位置。
- **随机删除**（Randomly Delete）：随机删除句子中的单词。
- **语法树结构替换**：通过语法树结构，精准地替换单词。
- **加噪**（NoiseMix） (https://github.com/noisemix/noisemix)：类似于图像领域的加噪，NoiseMix提供9种单词级别和2种句子级别的扰动来生成更多的句子，例如：这是一本很棒的书，但是他们的运送太慢了。->这是本很棒的书，但是运送太慢了。
- **情境增强**（Contextual Augmentation）：这种数据增强算法是用于文本分类任务的独立于域的数据扩充。通过用标签条件的双向语言模型预测的其他单词替换单词，可以增强监督数据集中的文本。
- **生成对抗网络**：利用生成对抗网络的方法来生成和原数据同分布的数据，来制造更多的数据。在自然语言处理领域，有很多关于生成对抗网络的工作：
  - Generating Text via Adversarial Training
  - GANS for Sequences of Discrete Elements with the Gumbel-softmax Distribution
  - SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient
- **回译技术**（Back Translation）：回译技术是NLP在机器翻译中经常使用的一个数据增强的方法。其本质就是快速产生一些翻译结果达到增加数据的目的。回译的方法可以增加文本数据的多样性，相比替换词来说，有时可以改变句法结构等，并保留语义信息。但是，回译的方法产生的数据严重依赖于翻译的质量。
- **扩句-缩句-句法**：先将句子压缩，得到句子的缩写，然后再扩写，通过这种方法生成的句子和原句子具有相似的结构，但是可能会带来语义信息的损失。
- **无监督数据扩增**（Unsupervised Data Augmentation）：通常的数据增强算法都是为有监督任务服务，这个方法是针对无监督学习任务进行数据增强的算法，UDA方法生成无监督数据与原始无监督数据具备分布的一致性，而以前的方法通常只是应用高斯噪声和Dropout噪声（无法保证一致性）。(https://arxiv.org/abs/1904.12848)



# 论文笔记

论文集合：https://tech.meituan.com/2021/08/05/acl-2021-meituan-07-papers.html

#### 关于知识图的调查：表示、获取与应用

**AAAI 2020** 的关于**知识图谱的最新权威综述论文**

全面的研究主题

1. 知识图表示学习 **KRL**:
   表示空间、评分函数、编码模型和辅助信息

   ```
   Q. Wang, Z. Mao, B. Wang, and L. Guo, “Knowledge graph embedding:
   A survey of approaches and applications,” IEEE TKDE, vol. 29, no. 12,
   pp. 2724–2743, 2017.
   
   Y. Lin, X. Han, R. Xie, Z. Liu, and M. Sun, “Knowledge representation
   learning: A quantitative review,” arXiv preprint arXiv:1812.10901, 2018.
   ```

   

2. 知识获取与知识图谱补全 **graph completion**
   知识获取：知识图嵌入 **KGE**(embedding)、路径推理、逻辑规则推理
   实体获取任务分为实体识别、输入、消歧和对齐；

3. 时间知识图

4. 知识感知应用

总结最近的突破和视角方向，以促进未来的研究。

1. 知识表示学习是知识图谱的一个关键研究问题

   1）表示空间，关系和实体在其中被表示

   2）用于衡量事实可信度的评分函数

   3）代表和学习关系性互动的编码模型。

   4）要纳入嵌入方法的辅助信息。

#### AAAI17 将知识图谱嵌入到主题模型

Incorporating Knowledge Graph Embeddings into Topic Modeling

LDA + 实体向量编码： 解决了传统概率主题模型，没有人类知识的缺点。  评估了他们方法的有效性。

概率主题模型：PLSA和潜在Dirichlet分配（LDA），无监督话题产生了**难以解释**的话题。

contributions：

1. 知识图嵌入LDA (KGE-LDA)
2. Gibbs抽样推理方法，可以正确处理知识图嵌入编码的知识。
3. 三个广泛使用的数据集的实验

von Mises-Fisher (vMF) distribution  圆正态分布





#### ACL20 一种用于短文本流聚类的在线语义增强Dirichlet模型

An Online Semantic-enhanced Dirichlet Model for Short Text Stream Clustering

##### 0 摘要：

由于短文本流具有无限长、稀疏数据表示和集群演化等特性，因此对其进行聚类是一项具有挑战性的任务。现有的方法通常以批处理的方式利用短文本流。然而，确定最佳批大小通常是一项困难的任务，因为我们没有主题演进时的先验知识。此外，图形模型中传统的独立单词表示方式在短文本聚类中容易造成“术语歧义”问题。因此，本文提出了一种基于在线语义增强的Dirichlet模型(OSDM)用于短文本流聚类，该模型将出现词的语义信息(即上下文)集成到一个新的图形模型中，并对每个短文本在线自动聚类。大量的结果表明，与许多最先进的算法相比，OSDM在合成和现实数据集上具有更好的性能。

##### 1 介绍：

微博、Twitter和Facebook等在线社交平台不断生成大量的短文本数据。近年来，由于事件跟踪、热点话题检测和新闻推荐等许多现实应用，此类短文本流的聚类得到了越来越多的关注(Hadifar等人，2019)。然而，由于短文本流具有无限长、模式演化、数据表示稀疏等独特特性，短文本流的聚类仍然是一大挑战。

在过去的十年中，人们从不同的角度提出了许多方法来解决文本流聚类问题，每种方法都有各自的优缺点。最初，对静态数据的传统聚类算法进行了增强，并对文本流进行了转换(Zhong,2005)。很快，它们就被基于模型的算法所取代，如LDA  (Blei等人，2003年)、DTM (Blei和Lafferty,2006年)、TDPM  (Ahmed和Xing,2008年)、GSDMM(Yin和Wang等人，2016b)、DPMFP (Huang等人，2013年)、TM-LDA  (Wang等人，2012年)、NPMM (Chen等人，2019年)和MStream  (Yin等人，2018年)等等。然而，对于大多数已建立的方法，它们通常以批处理的方式工作，并假定批处理中的实例是可互换的。这一假设通常不适用于主题演进的文本数据语料库。对于不同的文本流，确定最佳批处理大小也是一项重要的任务。

此外，与长文本文档不同的是，短文本聚类还缺乏用于捕获语义的支持性术语出现(Gong et al.，2018)。对于现有的大多数短文本聚类算法，如Sumblr  (Shou et al.，2013)、DCT (Liang et al.，2016)和MStreamF (Yin et al.，  2018)，在它们的聚类模型中利用独立单词表示往往会导致歧义。

> 经常吃苹果可以改善你的健康和肌肉耐力。
>
> ”早餐建议喝一杯新鲜苹果汁。
>
> “新的苹果手表可以监控你的健康状况。
>
> 苹果将于今年12月推出新的智能手机iPhoneX

这两个主题的推文很少有共同的术语，比如“健康”或“苹果”。如果模型只处理单个术语表示来计算相似度，就会产生歧义。但是，共同出现的术语表示(即上下文)帮助模型正确地识别主题1。

为了解决上述问题，我们提出了一种基于在线语义增强的dirichlet模型的短文本流聚类方法。与现有的方法相比，它有以下优点。
(1)它允许以在线方式处理每一个到达的短文本。在线模型不仅不需要确定最优的批处理大小，而且有利于有效地处理大规模数据流;
(2)据我们所知，在基于模型的在线聚类中首次集成语义信息，能够有效地处理“术语歧义”问题，最终支持高质量的聚类;
(3)采用Poly  Urn Scheme，在我们的聚类模型中自动确定聚类(主题)的数量。

##### 2 相关工作

在过去的十年中，出现了许多文本流聚类算法。在这里，由于篇幅的限制，我们只报告了一些与我们的工作高度相关的基于模型的方法。详情请参考综合调查，例如

文本聚类的早期经典尝试是Latent  Dirichlet Allocation (LDA) (Blei et al.，  2003)。但是，它不能处理文本流的时态数据。为此，人们提出了许多LDA变体来考虑文本流，如动态主题模型(DTM)  (Blei和Lafferty,2006)、动态混合模型(DMM) (Wei等人，2007)、时态LDA (TLDA) (Wang等人，2012)、流LDA  (S-LDA) (Amoualian等人，2016)、以及带有特征分区的dirichlet混合模型(DPMFP) (Zhao et al.，  2016)。

**这些模型假定每个文档都包含丰富的内容，因此它们不适合处理短文本流。**

之后，设计了基于Dirichlet多项式混合模型的动态聚类主题(DCT)模型，通过**给每个文档（document）分配一个指定的主题topic**来处理短文本流(Liang et al.，2016)。很快，GSDMM被提出用**塌陷吉布斯采样**扩展DMM来推断簇的数量(Yin and  Wang,2014)。然而，大多数这些模型都没有研究文本流中的演化的主题(集群)，在文本流中的主题数量通常会随着时间的推移而变化。

为了自动检测聚类数量，(Ahmed  and  Xing,2008)提出了一种时间dirichlet过程混合模型(TDMP)。它将文本流划分为许多块(批)，并假定每个批中的文档是可互换的。随后，提出了采用塌吉布斯采样的GSDPMM方法来推断每个批次的簇数。与LDA相比，GSDPMM不仅收敛速度更快，而且还能随时间动态分配集群数量(Yin  and Wang,2016a)。然而，TDMP和GSDPMM模型都不检查正在发展的主题，而且这些模型会多次处理文本流。随后，MStreamF (Yin et  al.，2018)被引入**遗忘机制**来应对集群进化，允许每批只处理一次。NPMM模型(Chen et  al.，2019)是最近引入的，它使用**单词嵌入**来消除模型的一个聚类生成参数。

总之，对于大多数现有的方法，它们通常以批处理的方式工作。然而，为不同的文本流确定最佳批处理大小通常是一项困难的任务。更重要的是，由于短文本数据固有的稀疏数据表示，在现有的方法中很少研究语义。实际上，需要仔细考虑它们以减少短文本聚类中的术语歧义。

##### 3 前言

在此，首先给出了问题的陈述，然后简要介绍了狄利克雷过程和波利亚坛子模型

###### 3.1 问题公式化

###### 3.2 狄利克雷过程

狄利克雷过程用非参数随机过程来模拟数据。它是从(基础)分布中抽取样本的过程，其中每个样本本身就是一个分布

###### 3.3 波利亚坛子模型

##### 4 建议的方法

LDA

<img src="${picture}/image-20211213141206391.png" alt="image-20211213141206391" style="zoom:50%;" />



<img src="${picture}/image-20211213141331811.png" alt="image-20211213141331811" style="zoom:67%;" />

##### 5 实验

at:https://github.com/JayKumarr/OSDM.

##### 6 结论

在本文中，我们提出了一种新的在线语义增强dirichlet模型用于短文本流聚类。与现有的方法相比，OSDM不需要指定批大小和动态进化的集群数量。它动态地将每个到达的文档分配到一个现有的集群，或者基于poly  urn方案生成一个新的集群中。更重要的是，OSDM试图在提出的图形表示模型中加入语义信息，以消除短文本聚类中的术语歧义问题。基于语义嵌入和在线学习，我们的方法可以找到高质量的进化集群。广泛的结果进一步证明，与许多先进的算法相比，OSDM具有更好的性能。



PPT

1. 

2.  Outline
   Motivation 
   Existing Problems
   Proposed Model
   Experimentations
   Conclusion

3.  Motivation

   Short-Text data generated by many online sources

4.  Previous Algorithms
   Similarity Based

   - HPStream
   - FW-Kmeans
   - ConStream

   Model Based

   - LDA
   - DTM
   - NPMM
   - MStream

5.  Challenges
   Semantic Information

   - Term ambiguity

     - context of a word change related to accompanied words
   - Concept Drift
     - Change in topic over time
  - Lift-span of topics differs
   - Batch vs Online 
     - Batch Processing assumption: no concept drift inside a batch

6. 

