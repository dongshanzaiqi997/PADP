import datetime
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from datetime import datetime
from Config import Config
from Model import ActorCritic
from StateModel import Environment
from Train import Train
from Test import Test
from tqdm import trange
from tensorboardX import SummaryWriter

def main():
    trainlabel = '2022-03-31T15-45-04'  # 2021-06-18T09-02-18
    state_model = Environment()

    if trainlabel == '':
        for j in range(1):
            ac = ActorCritic(Config.num_inputs-3, Config.num_control)
            #ac.load_net('2021-06-05T16-03-04')
            TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
            writer = SummaryWriter('logs/train/' + TIMESTAMP)
            writer.add_text('parameters/Ps', str(Config.safe_prob))
            test = Test(writer)
            train = Train(writer)

            for i in trange(Config.num_iteration):
                train.iteration_index = i
                if i % Config.evaluate_frequency == 0:
                    if i > 0:
                        #test.evaluate_train(ac, state_model)
                        #test.evaluate_nn_dis(ac, state_model, 0.9, 0.999, train)
                        test.render_episode(ac, state_model)
                        ac.save_net(TIMESTAMP)
                        #train.MC_safe_prob(ac, state_model)

                train.next_state(ac, state_model)

                # if i % 10 == 0:
                #     train.clean_accu_rew()

                #train.MC_safe_prob(ac, state_model)
                #train.est_safe_prob(ac, state_model)
                train.PEV(ac, state_model)

                train.accu_rew()
                train.PF()

                train.PIM(ac)
                train.x = train.x_next.clone()

            ac.save_net(TIMESTAMP)
            state_model.close()

    else:
        ac = ActorCritic(Config.num_inputs-3, Config.num_control)
        test = Test([])
        ac.load_net(trainlabel)  #2021-04-06T14-56-140.999  2021-04-06T14-45-570.9
        test.render_episode(ac, state_model)
    #test.evaluate_nn(ac, state_model, bad_episode=not True)
    #test.plot_nn()

    #plt.show()
    
    
if __name__ == '__main__':
    main()
