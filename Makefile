.PHONY : data

data :
		rsync -avz pabbeel@primus.banatao.berkeley.edu:data/reinforcement-lfd/ data/
