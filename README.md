# 2019_MCM_Contest_Problem_C_Meritorious_Winner
In 2019, we three students from SCU, department of Math and department of Economics participated into the MCM contest for Big Data Problem C.

  
题目解读
对美国五个州的毒品/成瘾性药品 的使用/滥用情况  建模。
数据part1主要分为对 dependent variable 因变量的描述： 按药品类别，州，县，年进行分类描述使用情况。
 
数据part2 是对不同细分地区的社会经济数据，算是额外的自变量。
要求就是先只用part 1的数据建模，然后引入part2的数据，最后结合出一个总模型评估影响因素和应对方案。

变量命名与假设：
对于美赛来说，变量命名是你开始建模的第一步，对问题里的所有相关因素命名会加深你对问题本身的理解。
而假设，则是用来弥补你的模型不能考虑到的部分，用假设将其补全。
 
变量命名，用好上下标是关键
数据可视化：
数据可视化是第一步，我们对需要预测的因变量往往需要给个详细的可视化和理解。
对于数量较多复杂的自变量，我们无需 每一个都可视化，往往是结合因变量计算一些简单的统计量，进行一个横向对比，经济学院同学熟知的有相关系数 Pearson 线性相关，对时间序列我们可以做格兰杰因果检验，对非线性模型我们可以用决策树或者decision tree里面的feature importance等等。
 
当然对多变量组合的检验和效果性分析我们也可以直接用机器学习模型train后在测试集上测试，多个变量的组合可以采用变量池添加变量法，变量池删除变量法 或者遗传算法局部搜索等等。
 
建模思路：
建模我们一般从较简单的基础模型开始，逐渐减少Assumption 增加模型复杂程度：
只看part 1的数据的话
最开始我们提出一个 SIS传染病模型，该通用模型对每个county都有效，我们对其微分：
 
当我们进一步考虑地区之间人口流动和扩散：我们就建立了一个微分方程组
 
对微分方程组的参数估计最简单的方法就是将其离散化为差分方程，进而代入历史数据，用最小二乘 对 参数进行估计。
 
  
加上part 2的大数据，我们用Gradient Boosted Decision Trees和 线性相关指标 来筛选变量importance，
 
GBDT的特性一是作为tree 本身，可以描述非线性关系，二是作为一个Boosted 模型，它容易过拟合。
所以我们最后将它和 之前 的回归模型结合使用，以求得到最好结果。
 
 
结论
 
 
最后我们对 之前 识别 出来的变量进行控制，并结合预测模型 对不同的变量未来的结果进行控制。
上面两张图 分别反应了 帮助成瘾者和投资本地教育的控制结结果。


