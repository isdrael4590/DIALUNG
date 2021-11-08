class DarkCovidNet(nn.Module):
    """
    Estructura de Red obtenida de https://www.sciencedirect.com/science/article/abs/pii/S0010482520301621
    """

    def __init__(self, numero_canales_entrada, clases_salida):
        super(DarkCovidNet, self).__init__()
        self.DN_1 = self.DN_block(numero_canales_entrada, 8)
        self.max_pool_1 = nn.MaxPool2d(2, stride=2)         
        self.DN_2 = self.DN_block(8, 16)
        self.max_pool_2 = nn.MaxPool2d(2, stride=2) 
        self.triple_conv_1 = self.triple_Conv_block(16, 32)
        self.max_pool_3 = nn.MaxPool2d(2, stride=2) 
        self.triple_conv_2 = self.triple_Conv_block(32, 64)
        self.max_pool_4 = nn.MaxPool2d(2, stride=2) 
        self.triple_conv_3 = self.triple_Conv_block(64, 128)
        self.max_pool_5 = nn.MaxPool2d(2, stride=2) 
        self.triple_conv_4 = self.triple_Conv_block(128, 256)
        self.DN_3 = self.DN_block(256, 128, size=1)
        self.DN_4 = self.DN_block(128, 256)
        self.conv = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False) 
        self.flatten = nn.Flatten()
        self.Linear =  nn.Linear(338, clases_salida) #4 clases

    def forward(self, x):
        """
        Recibe una imagen y ejecuta el entrenamiento
        """
        #Inicio del Feature Detector
        x = self.DN_1(x) #Primera operacion
        x = self.max_pool_1(x)
        x = self.DN_2(x) #Segunda operacion
        x = self.max_pool_2(x)
        x = self.triple_conv_1(x) #Tercera operacion
        x = self.max_pool_3(x)
        x = self.triple_conv_2(x) #Cuarta operacion
        x = self.max_pool_4(x)
        x = self.triple_conv_3(x) #Quinta operación
        x = self.max_pool_5(x)
        x = self.triple_conv_4(x) #Sexta operación
        x = self.DN_3(x) #Septima operación
        x = self.DN_4(x) #Octava operación
        x = self.conv(x) #Novena operación
        #Inicio del clasificador
        x = self.flatten(x)
        x = self.Linear(x)
        return x
    
    def DN_block(self, ni, nf, size = 3, stride = 1):
        """
        Definicion del bloque DN de acuerdo al Paper
        """
        for_pad = lambda s: s if s > 2 else 3
        return nn.Sequential(
            nn.Conv2d(ni, nf, kernel_size=size, stride=stride, padding=(for_pad(size) - 1)//2, bias=False), 
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)  
        )
    def triple_Conv_block(self, ni, nf):
        return nn.Sequential(
            self.DN_block(ni, nf),
            self.DN_block(nf, ni, size=1),  
            self.DN_block(ni, nf)
        )
 
