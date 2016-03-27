# coding: utf-8
import svgwrite as sw
from IPython import display
from IPython.display import SVG
import numpy as np
import time

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

np.random.seed(0)

# 過去何コマを見るか
STATE_NUM = 4

# DQN内部で使われるニューラルネット
class Q(Chain):
    def __init__(self,state_num=STATE_NUM):
        super(Q,self).__init__(
             l1=L.Linear(state_num, 16),  # stateがインプット
             l2=L.Linear(16, 64),
             l3=L.Linear(64, 256),
             l4=L.Linear(256, 1024),
             l5=L.Linear(1024, 2), # 出力2チャネル(Qvalue)がアウトプット
        )

    def __call__(self,x,t):
        return F.mean_squared_error(self.predict(x,train=True),t)

    def  predict(self,x,train=False):
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = F.leaky_relu(self.l3(h2))
        h4 =  F.leaky_relu(self.l4(h3))
        y = F.leaky_relu(self.l5(h4))
        return y

# DQNアルゴリズムにしたがって動作するエージェント
class DQNAgent():
    def __init__(self, epsilon=0.99):
        self.model = Q()
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.epsilon = epsilon # ランダムアクションを選ぶ確率
        self.actions=[-1,1] #　行動の選択肢
        self.experienceMemory = [] # 経験メモリ
        self.memSize = 300*100  # 経験メモリのサイズ(300サンプリングx100エピソード)
        self.experienceMemory_local=[] # 経験メモリ（エピソードローカル）
        self.memPos = 0 #メモリのインデックス
        self.batch_num = 32 # 学習に使うバッチサイズ
        self.gamma = 0.9       # 割引率
        self.loss=0
        self.total_reward_award=np.ones(100)*-1000 #100エピソード

    def get_action_value(self, seq):
        # seq後の行動価値を返す
        x = Variable(np.hstack([seq]).astype(np.float32).reshape((1,-1)))
        return self.model.predict(x).data[0]

    def get_greedy_action(self, seq):
        action_index = np.argmax(self.get_action_value(seq))
        return self.actions[action_index]

    def reduce_epsilon(self):
        self.epsilon-=1.0/100000

    def get_epsilon(self):
        return self.epsilon

    def get_action(self,seq,train):
        '''
        seq (theta, old_theta)に対して
        アクション（モータのトルク）を返す。
        '''
        action=0
        if train==True and np.random.random()<self.epsilon:
            # random
            action = np.random.choice(self.actions)
        else:
            # greedy
            action= self.get_greedy_action(seq)
        return action

    def experience_local(self,old_seq, action, reward, new_seq):
        #エピソードローカルな記憶
        self.experienceMemory_local.append( np.hstack([old_seq,action,reward,new_seq]) )

    def experience_global(self,total_reward):
        #グローバルな記憶
        #ベスト100に入る経験を取り込む
        if np.min(self.total_reward_award)<total_reward:
            i=np.argmin(self.total_reward_award)
            self.total_reward_award[i]=total_reward

            # GOOD EXPERIENCE REPLAY
            for x in self.experienceMemory_local:
                self.experience( x )

        #一定確率で優秀でないものも取り込む
        if np.random.random()<0.01:
            # # NORMAL EXPERIENCE REPLAY
            for x in self.experienceMemory_local:
                self.experience( x )

        self.experienceMemory_local=[]

    def experience(self,x):
        if len(self.experienceMemory)>self.memSize:
            self.experienceMemory[int(self.memPos%self.memSize)]=x
            self.memPos+=1
        else:
            self.experienceMemory.append( x )

    def update_model(self,old_seq, action, reward, new_seq):
        '''
        モデルを更新する
        '''
        # 経験メモリにたまってない場合は更新しない
        if len(self.experienceMemory)<self.batch_num:
            return

        # 経験メモリからバッチを作成
        memsize=len(self.experienceMemory)
        batch_index = list(np.random.randint(0,memsize,(self.batch_num)))
        batch =np.array( [self.experienceMemory[i] for i in batch_index ])
        x = Variable(batch[:,0:STATE_NUM].reshape( (self.batch_num,-1)).astype(np.float32))
        targets=self.model.predict(x).data.copy()

        for i in range(self.batch_num):
            #[ seq..., action, reward, seq_new]
            a = batch[i,STATE_NUM]
            r = batch[i, STATE_NUM+1]
            ai=int((a+1)/2) #±1 をindex(0,1)に。
            new_seq= batch[i,(STATE_NUM+2):(STATE_NUM*2+2)]
            targets[i,ai]=( r+ self.gamma * np.max(self.get_action_value(new_seq)))
        t = Variable(np.array(targets).reshape((self.batch_num,-1)).astype(np.float32)) 

        # ネットの更新
        self.model.zerograds()
        loss=self.model(x ,t)
        self.loss = loss.data
        loss.backward()
        self.optimizer.update()

# ずっと右って言い続けるお馬鹿なエージェント
class dummyAgent():
    def __init__(self):
        pass
    def get_action(self,seq,train):
        return 1.0
    def update_model(self,old_seq, action, reward, new_seq):
        pass
    def experience_local(self,old_seq, action, reward, new_seq):
        pass
    def experience_global(self):
        pass

# 手動でチューニングしたえらいエージェント。時間使ってるから反則。
class handmadeAgent():
    def __init__(self, rewind=18, accel=76, brake=77):
        self.time=0
        self.rewind=rewind
        self.accel=accel
        self.brake=brake

    def experience_local(self,old_seq, action, reward, new_seq):
        pass
    def experience_global(self):
        pass

    def get_action(self, seq,train):
        if self.time<self.rewind:
            out=-1
        elif self.time<self.accel:
            out=1
        elif self.time<self.brake:
            out=-1
        else:
            out=int((np.sign(seq[1]-seq[0])))
        self.time+=1
        return out

    def update_model(self,old_seq, action, reward, new_seq):
        pass


class pendulumEnvironment():
    '''
    振り子振り上げ環境。入力actionはモータのトルク、rewardはポール先端の高さ。
    '''
    def __init__(self):
        self.reset(0,0)

    def reset(self,initial_theta, initial_dtheta):
        self.th          = initial_theta
        self.th_old   = self.th
        self.th_ = initial_dtheta
        self.g=0.01
        self.highscore=-1.0

    def get_reward(self):
        '''
        高さプラスなら5倍ボーナスで高さに比例した正の報酬
        マイナスなら低さに比例した負の報酬
        '''
        reward=0
        h=-np.cos(self.th)
        if h>=0:
            reward= 5*np.abs(h)
        else:
            reward= -np.abs(h)
        return reward

    def get_state(self):
        return self.th

    def update_state(self, action):
        '''
        action はモータのトルク。符号しか意味を持たない。
        正なら0.005, 0なら0, 負なら-0.005
        '''
        power = 0.005* np.sign(action)

        self.th_ += -self.g*np.sin(self.th)+power
        self.th_old = self.th
        self.th += self.th_

    def get_svg(self):
        """
        アニメーション用に現在の状況をSVGで返す
        """
        dr=sw.Drawing("hoge.svg",(150,150))
        c=(75,75)
        dr.add(dr.line(c,(c[0]+50*np.sin(self.th),c[1]+50*np.cos(self.th)), stroke=sw.utils.rgb(0,0,0),stroke_width=3))
        return SVG(dr.tostring())

# 環境とエージェントを渡すとシミュレーションするシミュレータ。
# ここにシーケンスを持たせるのはなんか変な気もするけどまあいいか。。
class simulator:
    def __init__(self, environment, agent):
        self.agent = agent
        self.env = environment

        self.num_seq=STATE_NUM
        self.reset_seq()
        self.learning_rate=1.0
        self.highscore=0
        self.log=[]

    def reset_seq(self):
        self.seq=np.zeros(self.num_seq)

    def push_seq(self, state):
        self.seq[1:self.num_seq]=self.seq[0:self.num_seq-1]
        self.seq[0]=state

    def run(self, train=True, movie=False, enableLog=False):

        self.env.reset(0,0)

        self.reset_seq()
        total_reward=0

        for i in range(300):
            # 現在のstateからなるシーケンスを保存
            old_seq = self.seq.copy()

            # エージェントの行動を決める
            action = self.agent.get_action(old_seq,train)

            # 環境に行動を入力する
            self.env.update_state(action)
            reward=self.env.get_reward()
            total_reward +=reward

            # 結果を観測してstateとシーケンスを更新する
            state = self.env.get_state()
            self.push_seq(state)
            new_seq = self.seq.copy()

            # エピソードローカルなメモリに記憶する
            self.agent.experience_local(old_seq, action, reward, new_seq)

            if enableLog:
                self.log.append(np.hstack([old_seq[0], action, reward]))

            # 必要ならアニメを表示する
            if movie:
                display.clear_output(wait=True)
                display.display(self.env.get_svg())
                time.sleep(0.01)


        # エピソードローカルなメモリ内容をグローバルなメモリに移す
        self.agent.experience_global(total_reward)

        if train:
            # 学習用メモリを使ってモデルを更新する
            self.agent.update_model(old_seq, action, reward, new_seq)
            self.agent.reduce_epsilon()

        if enableLog:
            return total_reward,self.log

        return total_reward


if __name__ == '__main__':
    agent=DQNAgent()
    env=pendulumEnvironment()
    sim=simulator(env,agent)

    test_highscore=0

    fw=open("log.csv","w")

    for i in range(30000):
        total_reward=sim.run(train=True, movie=False)

        if i%1000 ==0:
            serializers.save_npz('model/%06d.model'%i, agent.model)

        if i%10 == 0:
            total_reward=sim.run(train=False, movie=False)
            if test_highscore<total_reward:
                print "highscore!",
                serializers.save_npz('model/%06d_hs.model'%i, agent.model)
                test_highscore=total_reward
            print i,
            print total_reward,
            print "epsilon:%2.2e" % agent.get_epsilon(),
            print "loss:%2.2e" % agent.loss,
            aw=agent.total_reward_award
            print "min:%d,max:%d" % (np.min(aw),np.max(aw))

            out="%d,%d,%2.2e,%2.2e,%d,%d\n" % (i,total_reward,agent.get_epsilon(),agent.loss, np.min(aw),np.max(aw))
            fw.write(out)
            fw.flush()
    fw.close
