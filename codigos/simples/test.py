from random import randint


def dad ():
    dado1 = randint (1,6)
    dado2 = randint (1,6)
    roll = 1
    while dado1 != dado2:
        print("{} {}".format(dado1,dado2))
        roll += 1
        dado1 = randint (1,6)
        dado2 = randint (1,6)
    print("{} {}".format(dado1,dado2))
    return roll
def dados ():
    dado1 = randint (1,6)
    dado2 = randint (1,6)
    roll = 1
    while dado1 != dado2:
        roll +=1
        dado1 = randint (1,6)
        dado2 = randint (1,6)
    return print ('o numero de rolagens até sair um valor igual foi: {}'.format(roll))


x = 0



(dados())