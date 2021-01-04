import numpy
import torch
import time

numpy.random.seed(0)


class model(torch.nn.Module):

    def __init__(self,layer_weights):
        torch.nn.Module.__init__(self)

        self._layers = []
        for i in range(len(layer_weights)):
            self._layers.append(
                torch.nn.Linear(layer_weights[i].shape[-1], layer_weights[i].shape[0],
                    bias=False))
            torch.nn.Module.add_module(self, f"layer{i}", self._layers[-1])
    def forward(self, x):

        for i in range(len(self._layers)):
            x = self._layers[i](x)
        return x

    def count_parameters(self):
        return numpy.sum([ numpy.prod(p.shape) for p in self.parameters() ] )

def compute_jacobian(ninput, nparameters, M, input_vector ):

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

    return jacobian

def main(ninput, n_filters_list, n_jacobian_calculations):

    cross_check_parameters = {}

    # Create an input vector:
    input_vector = numpy.random.random([ninput,1])

    cross_check_parameters['input_sum'] = numpy.sum(input_vector)
    cross_check_parameters['input_std'] = numpy.std(input_vector)

    n_filters_list.insert(0,1)
    n_filters_list.append(1)

    # This transpose is to match the tensorflow output!!
    # layer_1_weights = numpy.random.random([1,n_hidden_params]).T
    # # This reshape is because it is a different layout from tensorflow!
    # layer_2_weights = numpy.random.random([n_hidden_params,1]).reshape([1,n_hidden_params])

    layer_weights = [ numpy.random.random([n_filters_list[i],n_filters_list[i+1]]) for i in range(len(n_filters_list)-1)]
    layer_weights = [l.T for l in layer_weights]

    # Cast the input to torch:
    input_vector = torch.tensor(input_vector).float()

    if torch.cuda.is_available():
        input_vector = input_vector.cuda()

    # Create the model:
    M = model(layer_weights)

    # Switch out the layer weights for the controlled ones:
    new_dict = M.state_dict()
    for i, key in enumerate(new_dict.keys()):
        new_dict[key] = torch.tensor(layer_weights[i])
    M.load_state_dict(new_dict)

    if torch.cuda.is_available():
        M.cuda()

    # Forward pass:
    output = M(input_vector)

    # Capture the number of parameters:
    cross_check_parameters['n_params'] = M.count_parameters()
    nparameters = M.count_parameters()

    # Capture the network output:
    if torch.cuda.is_available():
        cross_check_parameters['output_sum'] = numpy.sum(output.cpu().detach().numpy())
        cross_check_parameters['output_std'] = numpy.std(output.cpu().detach().numpy())
    else:
        cross_check_parameters['output_sum'] = numpy.sum(output.detach().numpy())
        cross_check_parameters['output_std'] = numpy.std(output.detach().numpy())

    start = time.time()
    cross_check_parameters['jacobian_times'] = []
    for i in range(n_jacobian_calculations):
        this_start = time.time()
        jacobian = compute_jacobian(ninput, nparameters, M, input_vector)
        this_end = time.time()
        cross_check_parameters['jacobian_times'].append((this_end - this_start))


    end = time.time()


    end = time.time()
    cross_check_parameters['n_filters_list'] = n_filters_list
    cross_check_parameters['jacobian_sum']  = numpy.sum(jacobian.numpy())
    cross_check_parameters['jacobian_std']  = numpy.std(jacobian.numpy())
    cross_check_parameters['jacobian_prod'] = numpy.prod(jacobian.numpy())
    cross_check_parameters['jacobian_time'] = (end - start) / n_jacobian_calculations
    cross_check_parameters['jacobian_n_calls'] = n_jacobian_calculations

    return cross_check_parameters

if __name__ == '__main__':
    ninput = 24
    network_list = [
        [32, 32, 16],
        [128, 128],
        [512, 512, 512],
        [16, 16, 16, 16, 16, 16],
        [2048],
    ]
    for network in network_list:
        ccp = main(ninput, network, 5)
        print(ccp)
