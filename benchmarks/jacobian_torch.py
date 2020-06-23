import numpy
import torch
import time

numpy.random.seed(0)


class model(torch.nn.Module):

    def __init__(self,n_hidden_params):
        torch.nn.Module.__init__(self)

        self.layer1 = torch.nn.Linear(1, n_hidden_params, bias=False)
        self.layer2 = torch.nn.Linear(n_hidden_params, 1, bias=False)

    def forward(self, x):

        x = self.layer1(x)
        return self.layer2(x)


    def count_parameters(self):
        return numpy.sum([ numpy.prod(p.shape) for p in self.parameters() ] )


def main(ninput, n_hidden_params, n_jacobian_calculations):



    cross_check_parameters = {}
    
    # Create an input vector:
    input_vector = numpy.random.random([ninput,1])


    cross_check_parameters['input_sum'] = numpy.sum(input_vector)
    cross_check_parameters['input_std'] = numpy.std(input_vector)

    # This transpose is to match the tensorflow output!!
    layer_1_weights = numpy.random.random([1,n_hidden_params]).T
    # This reshape is because it is a different layout from tensorflow!
    layer_2_weights = numpy.random.random([n_hidden_params,1]).reshape([1,n_hidden_params])


    # Cast the input to torch:
    input_vector = torch.tensor(input_vector).float()

    # Create the model:
    M = model(n_hidden_params)

    # Switch out the layer weights for the controlled ones:
    new_dict = M.state_dict()
    new_dict['layer1.weight'] = torch.tensor(layer_1_weights)
    new_dict['layer2.weight'] = torch.tensor(layer_2_weights)
    M.load_state_dict(new_dict)

    # Forward pass:
    output = M(input_vector)
    
    # Capture the number of parameters:
    cross_check_parameters['n_params'] = M.count_parameters()
    nparameters = M.count_parameters()

    # Capture the network output:
    cross_check_parameters['output_sum'] = numpy.sum(output.detach().numpy())
    cross_check_parameters['output_std'] = numpy.std(output.detach().numpy())

    start = time.time()
    for i in range(n_jacobian_calculations):
        # Forward pass:
        output = M(input_vector)
    
        jacobian = torch.zeros(size=[ninput, nparameters])
        
        param_shapes = [p.shape for p in M.parameters() ]

        for n in range(ninput):
            output_n = output[n]
            M.zero_grad()
            params = M.parameters()
            do_dn = torch.autograd.grad(output_n, params, retain_graph=True)
            do_dn = torch.cat([g.flatten() for g in do_dn])
            jacobian[n,:] = torch.t(do_dn)

    end = time.time()
    cross_check_parameters['jacobian_sum']  = numpy.sum(jacobian.numpy())
    cross_check_parameters['jacobian_std']  = numpy.std(jacobian.numpy())
    cross_check_parameters['jacobian_prod'] = numpy.prod(jacobian.numpy())
    cross_check_parameters['jacobian_time'] = (end - start) / n_jacobian_calculations
    cross_check_parameters['jacobian_n_calls'] = n_jacobian_calculations

    return cross_check_parameters

if __name__ == '__main__':
    ccp = main(512, 128, 5)
    print(ccp)


