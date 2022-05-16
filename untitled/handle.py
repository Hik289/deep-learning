from models.BasicModule import BasicModule
import torch.nn as nn
import torch

from .t.bottleneck1d import bottleneck_1
from .t.bottleneck1d import bottleneck_2

from .t.LSTMcell_sxy import LSTMcell
from .t.GRUcell_sxy import GRUcell
from .t.RNNcell_sxy import RNNcell
from .t.LSTMCcell_sxy import LSTMCcell

from .t.PASScell_1_sxy import PASScell_1
from .t.PASScell_2_sxy import PASScell_2

##################################################################################################
# rnn_tradition
##################################################################################################
class rnn_sxy_2_7(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_7,self).__init__()
        self.model_name = 'rnn_sxy_2_7:daily'

        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 3)

        self.rnn2 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 3)

        self.linear1 = nn.Linear(250,30)
        self.linear2 = nn.Linear(250,30)

        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()


    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x, axis = 2)

        x_signal_day,_ = self.rnn1(x_day.permute(1,0,2))
        x_signal_bar,_ = self.rnn2(x[:,-1,:,:].permute(1,0,2))


        x_signal_day = self.linear1(x_signal_day)
        x_signal_bar = self.linear2(x_signal_bar)

        x_signal_2 = x_signal_bar[-1,:,:] + x_signal_day[-1,:,:]
        x_signal_2 = self.dropout(x_signal_2)
        # x_signal = self.linear(x_signal)
        x_signal_2 = self.act(x_signal_2)

        out = self.end_layer(x_signal_2)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal_2
        return self.model_out


class rnn_sxy_2_35(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_35,self).__init__()
        self.model_name = 'rnn_sxy_2_7:daily'

        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 3)

        self.rnn2 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 3)

        self.pointwise1 = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1)
        self.pointwise2 = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        self.linear1 = nn.Linear(250,30)
        self.linear2 = nn.Linear(250,30)

        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()


    def forward(self,x):
        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)

        x_signal_day,_ = self.rnn1(x_day.permute(1,0,2))
        x_signal_bar,_ = self.rnn2(x[:,-98:,:].permute(1,0,2))

        x_signal_day = x_signal_day[-5:,:,:]
        x_signal_bar = x_signal_bar[-5:,:,:]
        x_signal_day = self.pointwise1(x_signal_day.unsqueeze(0)).squeeze(1).squeeze(0)
        x_signal_bar = self.pointwise2(x_signal_bar.unsqueeze(0)).squeeze(1).squeeze(0)

        # x_signal_day = self.act1(x_signal_day)
        # x_signal_bar = self.act2(x_signal_bar)

        x_signal_day = self.linear1(x_signal_day)
        x_signal_bar = self.linear2(x_signal_bar)

        x_signal_2 = x_signal_bar+ x_signal_day
        
        x_signal_2 = self.dropout(x_signal_2)
        # x_signal = self.linear(x_signal)
        x_signal_2 = self.act(x_signal_2)

        out = self.end_layer(x_signal_2)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal_2
        return self.model_out


##################################################################################################
# rnn_cell
##################################################################################################

class rnn_sxy_2_29(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_29,self).__init__()
        self.model_name = 'rnn_sxy_2_29: LSTM+RNN+pointwise'

        self.lstm = LSTMcell(20,30)
        self.rnn = RNNcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda() 
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()      
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()             
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda() 
        self.rnn_h_t = self.rnn_h_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        rnn_out, self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_h_t)
        o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_h_t)
            o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out

class rnn_sxy_2_42(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_42,self).__init__()
        self.model_name = 'rnn_sxy_2_42: LSTMC+PASScell_2+pointwise'

        self.lstm = LSTMCcell(20,30)
        self.rnn = PASScell_2(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_o_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()   
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_o_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda()
        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_o_t = self.rnn_o_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        self.rnn_o_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_o_t, self.rnn_h_t)
        o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_c_t, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            self.rnn_o_t, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_o_t,self.rnn_h_t)
            o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out



##################################################################################################
# cnn1d
##################################################################################################
class cnn1d_sxy_8(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_8,self).__init__()
        self.model_name = 'cnn1d_sxy_8: -- deeplabV0'
        
        self.side1 = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=128, kernel_size=3),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=3))

        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5)

        self.side2 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=3))

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5)


        self.side3 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=3))   

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=5)

        self.side4 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
                                    nn.ReLU())   

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size= 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1)
        self.act5 = nn.ReLU()

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size= 1)
        self.act6 = nn.ReLU()

        self.avgpool = nn.AvgPool1d(kernel_size= 2)

        self.linear = nn.Linear(in_features= 256, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()

    def forward(self,x):

        x_side1 = self.side1(x[:,-17:,:].permute(0,2,1)).squeeze(-1)

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.act1(x_signal)
        x_signal = x_signal[:,:,3:]
        x_signal = self.avgpool1(x_signal)

        x_side2 = self.side2(x_signal[:,:,-17:]).squeeze(-1)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_side3 = self.side3(x_signal[:,:,-17:]).squeeze(-1)

        x_signal = self.cnn3(x_signal)
        x_signal = self.act3(x_signal)
        x_signal = x_signal[:,:,2:]
        x_signal = self.avgpool3(x_signal)

        x_side4 = self.side4(x_signal).squeeze(-1)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act4(x_signal)

        x_signal = self.cnn5(x_signal)
        x_signal = self.act5(x_signal)
        x_signal = self.dropout(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act6(x_signal)

        x_signal = self.avgpool(x_signal)

        x_signal = x_signal.squeeze(-1)

        x_signal = x_signal + x_side1 + x_side2 + x_side3 + x_side4

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_14(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_14,self).__init__()
        self.model_name = 'cnn1d_sxy_14: -- DeeplabV2:2'
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5)
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.act2 = nn.ReLU()

        self.bottleneck1 = bottleneck_1(input_dim=128, output_dim = 256)
        self.bottleneck2 = bottleneck_2(input_dim=256, output_dim=256)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features= 256, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.bottleneck1(x_signal)
        x_signal = self.bottleneck2(x_signal)

        x_signal = self.dropout(x_signal[:,:,-5:])

        x_signal = self.pointwise(x_signal.permute(0,2,1).unsqueeze(-1)).squeeze(1).squeeze(-1)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_15(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_15,self).__init__()
        self.model_name = 'cnn1d_sxy_15: -- DeeplabV2:FINAL'
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5)
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()

        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()

        self.bottleneck1 = bottleneck_1(input_dim=32, output_dim = 64)
        self.bottleneck2 = bottleneck_2(input_dim=64, output_dim=64)

        self.bottleneck3 = bottleneck_1(input_dim=64, output_dim = 128)
        self.bottleneck4 = bottleneck_2(input_dim=128, output_dim=128)

        self.assp_frac1 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,dilation=1)
        self.assp_frac2 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,dilation=3)
        self.assp_frac3 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,dilation=5)
        self.assp_frac4 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,dilation=7)


        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features= 256, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.bottleneck1(x_signal)
        x_signal = self.bottleneck2(x_signal)

        x_signal = self.bottleneck3(x_signal)
        x_signal = self.bottleneck4(x_signal)

        assp1 = self.assp_frac1(x_signal)
        assp2 =  self.assp_frac2(x_signal) 
        assp3 = self.assp_frac3(x_signal) 
        assp4 = self.assp_frac4(x_signal)
        assp1 = nn.functional.interpolate(assp1, size = 16)
        assp2 = nn.functional.interpolate(assp2, size = 16)
        assp3 = nn.functional.interpolate(assp3, size = 16)
        assp4 = nn.functional.interpolate(assp4, size = 16)
        x_signal = assp1 + assp2 + assp3 + assp4

        x_signal = self.dropout(x_signal[:,:,-5:])

        x_signal = self.pointwise(x_signal.permute(0,2,1).unsqueeze(-1)).squeeze(1).squeeze(-1)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out



##################################################################################################
# cnn2d
##################################################################################################
class cnn2d_sxy_6(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn2d_sxy_6,self).__init__()
        self.model_name = 'cnn2d_sxy_6: cnn_deepwise_deepwise_ordinarymax_pointwise_1'
        
        self.deepwise1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.deepwise2 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size= 3)
        self.deepwise_act = nn.ReLU()

        self.ordinary1 = nn.Conv2d(in_channels= 20, out_channels= 40, kernel_size=3)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size= 2)    

        self.pointwise = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1)
        self.pointwise_act = nn.ReLU()

        self.linear1 = nn.Linear(in_features= 8, out_features= 1)
        self.linear2 = nn.Linear(in_features= 40, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.deepwise1(x.unsqueeze(1)).squeeze(-1)
        x_signal = self.deepwise2(x_signal.unsqueeze(1))
        x_signal = self.deepwise_act(x_signal)

        x_signal = self.ordinary1(x_signal)
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)
        

        x_signal = x_signal[:,:,:,-5:]
        x_signal = self.pointwise(x_signal.permute(0,3,2,1)).squeeze(1)
        x_signal = self.pointwise_act(x_signal)

        x_signal = self.dropout(x_signal)
        
        x_signal = self.linear1(x_signal.permute(0,2,1)).squeeze(-1)
        x_signal = self.linear2(x_signal)

        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn2d_sxy_7(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn2d_sxy_7,self).__init__()
        self.model_name = 'cnn2d_sxy_7: cnn_deepwise_deepwise_ordinary_2_pointwise_1'
        
        self.deepwise1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.deepwise2 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size= 3)
        self.deepwise_act = nn.ReLU()

        self.ordinary1 = nn.Conv2d(in_channels= 20, out_channels= 40, kernel_size=3)
        self.act1 = nn.ReLU()

        self.ordinary2 = nn.Conv2d(in_channels= 40, out_channels= 80, kernel_size=3)
        self.act2 = nn.ReLU()

        
        self.pointwise = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1)
        self.pointwise_act = nn.ReLU()


        self.linear1 = nn.Linear(in_features= 14, out_features= 1)
        self.linear2 = nn.Linear(in_features= 80, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.deepwise1(x.unsqueeze(1)).squeeze(-1)
        x_signal = self.deepwise2(x_signal.unsqueeze(1))
        x_signal = self.deepwise_act(x_signal)

        x_signal = self.ordinary1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.ordinary2(x_signal)
        x_signal = self.act2(x_signal)

        x_signal = x_signal[:,:,:,-5:]
        x_signal = self.pointwise(x_signal.permute(0,3,2,1)).squeeze(1)
        x_signal = self.pointwise_act(x_signal)

        x_signal = self.dropout(x_signal)
        
        x_signal = self.linear1(x_signal.permute(0,2,1)).squeeze(-1)
        x_signal = self.linear2(x_signal)

        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn2d_sxy_8(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn2d_sxy_8,self).__init__()
        self.model_name = 'cnn2d_sxy_8: cnn_deepwise_deepwise_ordinary_3_pointwise_1'
        
        self.deepwise1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.deepwise2 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size= 3)
        self.deepwise_act = nn.ReLU()

        self.ordinary1 = nn.Conv2d(in_channels= 20, out_channels= 40, kernel_size=3)
        self.act1 = nn.ReLU()

        self.ordinary2 = nn.Conv2d(in_channels= 40, out_channels= 80, kernel_size=3)
        self.act2 = nn.ReLU()

        self.ordinary3 = nn.Conv2d(in_channels= 80, out_channels= 160, kernel_size=3)
        self.act3 = nn.ReLU()
        
        self.pointwise = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1)
        self.pointwise_act = nn.ReLU()


        self.linear1 = nn.Linear(in_features= 12, out_features= 1)
        self.linear2 = nn.Linear(in_features= 160, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.deepwise1(x.unsqueeze(1)).squeeze(-1)
        x_signal = self.deepwise2(x_signal.unsqueeze(1))
        x_signal = self.deepwise_act(x_signal)

        x_signal = self.ordinary1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.ordinary2(x_signal)
        x_signal = self.act2(x_signal)

        x_signal = self.ordinary3(x_signal)
        x_signal = self.act3(x_signal)

        x_signal = x_signal[:,:,:,-5:]
        x_signal = self.pointwise(x_signal.permute(0,3,2,1)).squeeze(1)
        x_signal = self.pointwise_act(x_signal)

        x_signal = self.dropout(x_signal)
        
        x_signal = self.linear1(x_signal.permute(0,2,1)).squeeze(-1)
        x_signal = self.linear2(x_signal)

        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


