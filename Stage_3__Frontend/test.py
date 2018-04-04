from neural_net import neural_net_keras
import numpy as np

if __name__=='__main__':
    nn = neural_net_keras("cluster_2",(2,10,1))

    # inp = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
    # out = np.array([[1],[0],[0],[1]], "float32")

    # nn.train_from_arr(inp,out,epochs=1000,batch_size=4)

    # nn.evaluate_from_arr(inp,out,batch_size=4)
    
    # print(np.array(nn.predict(inp)).round())
    
    # nn.add_to_training_data([100,0],[1])
    # nn.write_back_training_data()
    # print(nn.training_data_x)
    # print(nn.training_data_y)
    
    # nn.train(epochs=1000,batch_size=3)
    # nn.evaluate(batch_size=3)
    
    # print(np.array(nn.predict(np.array([[1,0]]))))