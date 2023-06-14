from random import shuffle
import numpy as np
import matplotlib.pyplot as plt

global POP_SIZE,CROSS_RATE,MUTATION_RATE,N_GENERATIONS,TIME,M,N,MIN_TIME

TIME=[[31,41,25,30],[19,55,3,34],[23,42,27,6],[13,22,14,13],[33,5,57,19]]  #用时矩阵，子列表代表工件，子列表中的值代表对应工序用时
CROSS_RATE = 0.6
MUTATION_RATE = 0.1
N_GENERATIONS = 50
POP_SIZE = 50
N=len(TIME)
M=len(TIME[0])
MIN_TIME=0
for gongxu in range(M):
    max_=TIME[gongxu][0]
    for yuanjian in range(N):
        max_=TIME[yuanjian][gongxu] if TIME[yuanjian][gongxu]>max_ else max_
    MIN_TIME+=max_


def c(i,k,pred:list):   #递归求解一种方案的用时
    result=0
    if i>0 and k>0:
        result=max(c(i-1,k,pred),c(i,k-1,pred))+TIME[pred[i]][k]
    else:
        if i==0 and k>0:
            result=c(0,k-1,pred)+TIME[pred[0]][k]
        elif k==0 and i>0:
            result=c(i-1,0,pred)+TIME[pred[i]][0]
        else:
            result=TIME[pred[0]][0]
    return result

def get_time(pred):
    return c(N-1,M-1,pred=pred)

def get_fitness(pred):  #pred 是一个长度为N的列表[0,1,2,3]
    return 1/(get_time(pred)-MIN_TIME)

def select(pop, fitness):    # nature selection wrt pop's fitness
    p=[]
    sumup=sum(fitness)
    for i in fitness:
        p.append(i/sumup)
    # print(p)
    idx = np.random.choice(POP_SIZE, size=POP_SIZE, replace=True,p=p)
    new_pop=[]
    for i in idx:
        new_pop.append(pop[i].copy())
    return new_pop.copy()

def crossover(pop):     # 用奇数位和偶数位进行交叉
    offspring_list=pop.copy()
    fix_num=int(N/2)
    for m in range(int(POP_SIZE/2)):
        if CROSS_RATE>=np.random.rand():
            parent_1= pop[2*m].copy()    #奇数位的父DNA
            parent_2= pop[2*m+1].copy()  #偶数位的父DNA
            child_1=['' for i in range(N)]
            child_2=['' for i in range(N)]
            choice_1=np.random.choice(N,fix_num,replace=False)  #两位的索引数列
            choice_2=np.random.choice(N,fix_num,replace=False)  #两位的索引数列
            # print('choice_1:',choice_1,'choice_2:',choice_2)
            c1,c2=[],[]
            for a in choice_1:
                c1.append(parent_1[a])
            for b in choice_2:
                c2.append(parent_2[b])
            # print('c1:',c1,'c2:',c2)
            for i in range(N):
                if i in choice_1:
                    child_1[i]=parent_1[i]
                if i in choice_2:
                    child_2[i]=parent_2[i]

            for j in parent_2:
                if j not in child_1:
                    child_1[child_1.index('')]=j
                else:
                    continue
            for k in parent_1:
                if k not in child_2:
                    child_2[child_2.index('')]=k
                else:
                    continue
            # print('child_1:',child_1,'child_2:',child_2)
            offspring_list[2*m]=child_1.copy()
            offspring_list[2*m+1]=child_2.copy()
    return offspring_list.copy()

def mutate(child):
    new_child=child.copy()
    out_index=[]
    for point in range(N):
        if np.random.rand() < MUTATION_RATE:
            out_index.append(point)   #将要发生变异的值的索引列表
    choice=out_index.copy()
    shuffle(choice)
    for i in range(len(choice)):
        new_child[out_index[i]]=child[choice[i]]
    return new_child.copy()

def draw(f:list):
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(label='遗传算法FSP统计')
    plt.grid(axis='both')
    x=[0,N_GENERATIONS]
    c = 0.5 * (min(x) + max(x))
    d = max(f[0][1])
    plt.text(c, d, '交叉概率：'+str(CROSS_RATE), ha='center', fontsize=10, alpha=0.8)
    plt.text(c, d-5, '变异概率：'+str(MUTATION_RATE), ha='center', fontsize=10, alpha=0.8)
    plt.text(c, d-10, '群体规模：'+str(POP_SIZE), ha='center', fontsize=10, alpha=0.8)
    for i in range(N_GENERATIONS):
        x=[i for j in range(POP_SIZE)]
        plt.scatter(x,f[i][1],label='个体值',s=1,c='c')
        plt.pause(0.1)
        plt.scatter(i,f[i][2],label='最优值',s=20,c='r')
        plt.scatter(i,f[i][3],label='最劣值',s=20,c='g')
        plt.scatter(i,f[i][4],label='平均值',s=20,c='b')
        if i==0:plt.legend(loc='upper right')
    plt.xlabel('迭代次数:   '+str(N_GENERATIONS))
    plt.ylabel('方案用时(秒)')
    plt.xticks(size=9,rotation=30)
    plt.ioff()
    plt.show()
    return

if __name__=="__main__":
    pop,f=[],[]
    for i in range(POP_SIZE):
        random_num=list(np.random.permutation(N)) # generate a random permutation of 0 to N-1
        pop.append(random_num) # add to the pop
    for _ in range(N_GENERATIONS):
        # 画图列表结构：迭代次数、个体用时（长为POP_SIZE列表spend）、最低用时、最高用时、平均用时
        fitness,spend=[],[]
        for i in pop:
            fitness.append(get_fitness(i.copy()))
            spend.append(get_time(i.copy()))
        f.append([_,spend,min(spend),max(spend),sum(spend)/POP_SIZE])
        most_fitted=pop[fitness.index(max(fitness))]
        print(_,"Most fitted DNA: ",most_fitted,get_time(most_fitted),pop.count(most_fitted))
        pop = select(pop.copy(), fitness.copy())
        pop_copy = pop.copy()
        pop = crossover(pop_copy).copy()
        for parent in pop:
            child = mutate(parent)
            parent[:] = child.copy()
    draw(f=f)
    pass