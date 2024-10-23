import numpy as np

# lectura del archivo

# funcion para leer un archivo txt
def leer_datos(archivo):
    with open(archivo, 'r') as f:
        lineas = f.readlines()

    datos = [list(map(float, linea.strip().split())) for linea in lineas]

    return np.array(datos)

#normalizar los valores a la escala de 0 y 1
def normalizar_datos(datos):
    return datos / 3.0

#ejecucion
# se define el archivo y se llama a la función que lo abre
archivo = "datosMLP.txt"
datos = leer_datos(archivo)

# se llama a la función para normalizar los datos
datos_normalizados = normalizar_datos(datos)

# preparar datos y arquitectura
dias_semana = np.arange(0,7)
horas_dia = np.arange(1,25)

#combinaciones
entradas = []
salidas = []

for hora in range(24):
    for dia in range(7):
        entradas.append([hora, dia])
        salidas.append(datos_normalizados[hora, dia])

entradas = np.array(entradas)
salidas = np.array(salidas).reshape(-1, 1)

# definicion de la clase

class PerceptronMLP:
    def __init__(self, estructura):
        self.pesos = []
        self.sesgos = []
        self.activaciones = []

        for i in range(len(estructura)-1):
            peso = np.random.rand(estructura[i], estructura[i+1]) - 0.5
            sesgo = np.random.rand(1,estructura[i+1]) - 0.5
            self.pesos.append(peso)
            self.sesgos.append(sesgo)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivada(self, x):
        return x * (1-x)
    
    def propagacion(self, entrada):
        self.activaciones = [entrada]
        for peso, sesgo in zip(self.pesos, self.sesgos):
            z = np.dot(self.activaciones[-1], peso) + sesgo
            a = self.sigmoid(z)
            self.activaciones.append(a)
        return self.activaciones[-1]
    
    def retropropagacion(self, salida_esperada, tasa_aprendizaje):
        errores = [salida_esperada - self.activaciones[-1]]
        for i in reversed(range(len(self.pesos))):
            error = errores[-1]
            delta = error * self.sigmoid_derivada(self.activaciones[i+1])
            errores.append(np.dot(delta, self.pesos[i].T))
            self.pesos[i] += np.dot(self.activaciones[i].T, delta) * tasa_aprendizaje
            self.sesgos[i] += np.sum(delta, axis = 0, keepdims=True) * tasa_aprendizaje
        errores.reverse()
        return errores
    
    def entrenar(self, datos_entrada, datos_salida, iteraciones, tasa_aprendizaje):
        for _ in range(iteraciones):
            salida_predicha = self.propagacion(datos_entrada)
            self.retropropagacion(datos_salida, tasa_aprendizaje)


#inicio de las pruebas
# definir la estructura
estructura = [2,10,10,1]
perceptron = PerceptronMLP(estructura)

#entrenar la red
perceptron.entrenar(entradas, salidas, iteraciones=10000, tasa_aprendizaje=0.01)

#proponer una salida esperada

dia = 3
hora = 15
prediccion = perceptron.retropropagacion(np.array([[hora, dia]]))
print(f"El consumo para el día {dia} a las {hora}:00: {prediccion[0][0]}")