# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import collections
import time
import requests

def LineNotify(message):
    line_notify_token = 'yourtoken'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}  # 発行したトークン
    line_notify = requests.post(line_notify_api, data=payload, headers=headers,timeout=10)

class Environment:
	"""docstring for Enveronment"""
	data = 'global'
	posi = {}
	actions = []
	def __init__(self):
		self.data = pd.read_csv('summary.csv')
		self.posi['side'] = 0 # 0:no 1:BUY 2:SELL
		self.posi['size'] = 0
		self.posi['price'] = 0
		self.posi['pnl'] = 0

		self.actions = [[]]#actions[0]:ポジ全決算
		ask_spread = list(range(10,250,10))
		ask_spread.append(10000)
		bid_spread = list(range(10,250,10))
		bid_spread.append(10000)
		for a_s in ask_spread:
			for b_s in bid_spread:
				self.actions.append([a_s,b_s])

	def reset(self):
		observation = [0,0,0,0]#######################
		return observation

	def action_space(self):
		return self.actions

	def step(self,index,action):
		done = False
		observation = [0,0,0,0]####################################
		reward = 0
		nari_cost = -200

		if action == 0:
			if self.posi['side'] == 0:
				pass
			elif self.posi['side'] == 1:
				reward = self.data.iloc[index]['open']*self.posi['size'] - self.posi['price'] + nari_cost*self.posi['size'] 
				self.posi['side'] = 0
				self.posi['size'] = 0
				self.posi['price'] = 0
			elif self.posi['side'] == 2:
				reward = self.posi['price'] - self.data.iloc[index]['open']*self.posi['size'] + nari_cost*self.posi['size'] 
				self.posi['side'] = 0
				self.posi['size'] = 0
				self.posi['price'] = 0

		else:
			a_s,b_s = self.actions[action]

			if self.data.iloc[index]['open'] - b_s >= self.data.iloc[index]['low']:#買い約定
			
				if self.posi['side'] == 0:
					self.posi['side'] = 1
					self.posi['size'] = 1
					self.posi['price'] = self.data.iloc[index]['open'] - b_s
				elif self.posi['side'] == 1:					
					self.posi['size'] += 1
					self.posi['price'] += self.data.iloc[index]['open'] - b_s
				elif self.posi['side'] == 2:
					reward = self.posi['price'] - (self.data.iloc[index]['open'] - b_s)*self.posi['size']
					self.posi['side'] = 0
					self.posi['size'] = 0
					self.posi['price'] = 0

			if self.data.iloc[index]['open'] + a_s <= self.data.iloc[index]['high']:#売り約定

				if self.posi['side'] == 0:
					self.posi['side'] = 2
					self.posi['size'] = 1
					self.posi['price'] = self.data.iloc[index]['open'] + a_s
				elif self.posi['side'] == 1:					
					reward = (self.data.iloc[index]['open'] + a_s)*self.posi['size'] - self.posi['price']
					self.posi['side'] = 0
					self.posi['size'] = 0
					self.posi['price'] = 0
				elif self.posi['side'] == 2:
					self.posi['size'] += 1
					self.posi['price'] += self.data.iloc[index]['open'] + a_s
		if self.posi['side'] == 0:
			self.posi['pnl'] = 0
		elif self.posi['side'] == 1:
			self.posi['pnl'] = self.data.iloc[index]['close']*self.posi['size'] - self.posi['price']
		elif self.posi['side'] == 2:
			self.posi['pnl'] = self.posi['price'] - self.data.iloc[index]['close']*self.posi['size']

		if self.posi['size'] >= 4:
			reward += -10000

		if self.posi['pnl'] < -1000:
			reward += -10000
		observation = [self.posi['side'],
						self.posi['size'],
						self.data.iloc[index]['size'],
						self.posi['pnl']]
		if index >= len(self.data)-1:
			done = True
		return observation,reward,done

def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

def digitize_state(observation):
    # 各値をn個の離散値に変換
    n = 10
    side,size,volume,pnl = observation###################################################
    digitized = [np.digitize(side, bins=bins(0, 2, n)),
                 np.digitize(size,bins=bins(0, 5, n)),
                 np.digitize(volume,bins=bins(-50, 50, n)),
                 np.digitize(pnl,bins=bins(-1000, 1000, n))]
    # 変換
    return sum([x * (n ** i) for i, x in enumerate(digitized)])

def get_action(state, action, observation, reward, episode, q_table,env):
	next_state = digitize_state(observation)
	next_action = np.argmax(q_table[next_state])
	epsilon = 0.5 * (0.99 ** episode)
	if  epsilon <= np.random.uniform(0, 1):
		next_action = np.argmax(q_table[next_state])
	else:
		next_action = np.random.choice(list(range(len(env.action_space()))))###################################################

    # Qテーブルの更新
	alpha = 0.2
	gamma = 0.99
	q_table[state, action] = (1 - alpha) * q_table[state, action] +\
            alpha * (reward + gamma * q_table[next_state, next_action])

	return next_action, next_state

def main():
	LineNotify('シミュレーション始まったで')
	# q_table = np.loadtxt('out.csv',delimiter=',')
	env = Environment()
	n = 10 #離散値の分割数
	q_table = np.random.uniform(low=-1, high=1, size=(n **len(env.reset()), len(env.action_space())))##############################################
	num_episodes = 1000
	results = []
	results_full = []
	for episode in range(num_episodes):
		start_time = time.time()
		# 環境の初期化
		env = Environment()
		observation = env.reset()
		index = 0
		done = False
		rewards = []
		action_history = []
		mysize_history = []
		pnl_history = []
		state = digitize_state(observation)
		action = np.argmax(q_table[state])

		while not done:
			action_history.append(action)
			# 行動の実行とフィードバックの取得
			observation,reward,done = env.step(index,action)
			mysize_history.append(observation[1])
			pnl_history.append(observation[3])
			# 行動の選択
			action, state = get_action(state, action, observation, reward, episode, q_table,env)
			rewards.append(reward)

			index += 1

		rewards = np.cumsum(rewards)
		plt.figure()
		plt.plot(rewards)
		plt.grid(which='major',color='black',linestyle='-')
		plt.grid(which='minor',color='black',linestyle='-')
		plt.ylim(-300000,1000000)
		plt.title('maxsize:'+str(max(mysize_history))+' reward:'+str(rewards[-1]))
		plt.savefig('./figure/'+str(episode)+'.png')
		
		end_time = time.time()

		results.append(rewards[-1])
		results_full.append(rewards[-1])
		print(episode,sum(results)/len(results),'\n',max(mysize_history),'\n',end_time-start_time,"秒")#,collections.Counter(action_history))
		if len(results) > 50:
			del results[0]

	np.savetxt('out.csv',action_history,delimiter=',')
	np.savetxt('q_table.csv',q_table,delimiter=',')
	np.savetxt('action_history.csv',action_history,delimiter=',')
	np.savetxt('pnl_history.csv',pnl_history,delimiter=',')
	import dill
	dill.dump_session('session.pkl')
	
	plt.figure()
	plt.plot(results_full)
	plt.grid(which='major',color='black',linestyle='-')
	plt.grid(which='minor',color='black',linestyle='-')
	plt.savefig('figure.png')

	LineNotify('シミュレーション終わったで')

if __name__ == '__main__':
    main()