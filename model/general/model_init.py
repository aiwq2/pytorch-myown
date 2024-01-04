import torch.nn as nn

def xavier_init_model(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_normal_(m.weight)
        # nn.init.xavier_uniform_(m.weight)


# Initialize the parameters with zeros
def zero_init_model(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.constant_(m.weight,0.0)
        # nn.init.constant_(m.bias,0.0)
        # nn.init.zeros_(m.weight)

# Initialize the parameters with Gaussian random values
def normal_init_model(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.normal_(m.weight,mean=0.0,std=0.01)
        # nn.init.normal_(m.bias,mean=0.0,std=0.01)

# Initialize the parameters with He random values
def kaiming_init_model(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.kaiming_normal_(m.weight)
        # nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')  # Initialize weights with He random values
        # nn.init.kaiming_normal_(m.bias)



