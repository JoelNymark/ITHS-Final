import numpy as np
 
class Q_Learning:
    def __init__(self,env,alpha,gamma,epsilon,numberEpisodes,numberOfBins,lowerBounds,upperBounds):
        import numpy as np
        self.env=env
        self.alpha=alpha
        self.gamma=gamma 
        self.epsilon=epsilon 
        self.actionNumber=env.action_space.n 
        self.numberEpisodes=numberEpisodes
        self.numberOfBins=numberOfBins
        self.lowerBounds=lowerBounds
        self.upperBounds=upperBounds
        self.sumRewardsEpisode=[]
        self.Qmatrix=np.random.uniform(low=0, high=1, size=(numberOfBins[0],numberOfBins[1],numberOfBins[2],numberOfBins[3],self.actionNumber))

    def returnIndexState(self,state):
        position =          state[0]
        velocity =          state[1]
        angle    =          state[2]
        angularVelocity=    state[3]
         
        cartPositionBin=np.linspace(self.lowerBounds[0],self.upperBounds[0],self.numberOfBins[0])
        cartVelocityBin=np.linspace(self.lowerBounds[1],self.upperBounds[1],self.numberOfBins[1])
        poleAngleBin=np.linspace(self.lowerBounds[2],self.upperBounds[2],self.numberOfBins[2])
        poleAngleVelocityBin=np.linspace(self.lowerBounds[3],self.upperBounds[3],self.numberOfBins[3])
         
        indexPosition=np.maximum(np.digitize(state[0],cartPositionBin)-1,0)
        indexVelocity=np.maximum(np.digitize(state[1],cartVelocityBin)-1,0)
        indexAngle=np.maximum(np.digitize(state[2],poleAngleBin)-1,0)
        indexAngularVelocity=np.maximum(np.digitize(state[3],poleAngleVelocityBin)-1,0)
         
        return tuple([indexPosition,indexVelocity,indexAngle,indexAngularVelocity])   

    def selectAction(self,state,index):
         
        # first 500 episodes we select completely random actions to have enough exploration
        if index<500:
            return np.random.choice(self.actionNumber)   
             
        randomNumber=np.random.random()
         
        # after 7000 episodes, we slowly start to decrease the epsilon parameter
        if index>7000:
            self.epsilon=0.999*self.epsilon
        if randomNumber < self.epsilon:
            return np.random.choice(self.actionNumber)            
        else:
            return np.random.choice(np.where(self.Qmatrix[self.returnIndexState(state)]==np.max(self.Qmatrix[self.returnIndexState(state)]))[0])

    def simulateEpisodes(self):
        import numpy as np
        for indexEpisode in range(self.numberEpisodes):
            rewardsEpisode=[]
            (stateS,_)=self.env.reset()
            stateS=list(stateS)
            print("Simulating episode {}".format(indexEpisode))
            terminalState=False
            while not terminalState:
                stateSIndex=self.returnIndexState(stateS)
                actionA = self.selectAction(stateS,indexEpisode)
                (stateSprime, reward, terminalState,_,_) = self.env.step(actionA)          

                rewardsEpisode.append(reward)
                stateSprime=list(stateSprime)
                stateSprimeIndex=self.returnIndexState(stateSprime)
                QmaxPrime=np.max(self.Qmatrix[stateSprimeIndex])
                if not terminalState:
                    error=reward+self.gamma*QmaxPrime-self.Qmatrix[stateSIndex+(actionA,)]
                    self.Qmatrix[stateSIndex+(actionA,)]=self.Qmatrix[stateSIndex+(actionA,)]+self.alpha*error
                else:
                    error=reward-self.Qmatrix[stateSIndex+(actionA,)]
                    self.Qmatrix[stateSIndex+(actionA,)]=self.Qmatrix[stateSIndex+(actionA,)]+self.alpha*error
                stateS=stateSprime
            print("Sum of rewards {}".format(np.sum(rewardsEpisode)))        
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode))


    def simulateLearnedStrategy(self):
        import gym 
        import time
        env1=gym.make('CartPole-v1',render_mode='human')
        (currentState,_)=env1.reset()
        env1.render()
        timeSteps=1000
        obtainedRewards=[]
         
        for timeIndex in range(timeSteps):
            print(timeIndex)
            actionInStateS=np.random.choice(np.where(self.Qmatrix[self.returnIndexState(currentState)]==np.max(self.Qmatrix[self.returnIndexState(currentState)]))[0])
            currentState, reward, terminated, truncated, info =env1.step(actionInStateS)
            obtainedRewards.append(reward)   
            time.sleep(0.05)
            if (terminated):
                time.sleep(1)
                break
        return obtainedRewards,env1

    def simulateRandomStrategy(self):
        import gym 
        import time
        import numpy as np
        env2=gym.make('CartPole-v1')
        (currentState,_)=env2.reset()
        env2.render()
        episodeNumber=100
        timeSteps=1000
        sumRewardsEpisodes=[]
         
         
        for episodeIndex in range(episodeNumber):
            rewardsSingleEpisode=[]
            initial_state=env2.reset()
            print(episodeIndex)
            for timeIndex in range(timeSteps):
                random_action=env2.action_space.sample()
                observation, reward, terminated, truncated, info =env2.step(random_action)
                rewardsSingleEpisode.append(reward)
                if (terminated):
                    break     
            sumRewardsEpisodes.append(np.sum(rewardsSingleEpisode))
        return sumRewardsEpisodes,env2