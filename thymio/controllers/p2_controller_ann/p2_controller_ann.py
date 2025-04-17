import numpy as np
from controller import Supervisor

# Rede neuronal com 2 entradas, 1 camada escondida (4 neurónios), 2 saídas
class SimpleANN:
    def __init__(self, genome):
        self.input_size = 2
        self.hidden_size = 4
        self.output_size = 2

        # Extrair pesos e biases do genoma (tamanho total: 22)
        i = 0
        self.W1 = np.array(genome[i:i+8]).reshape((2, 4))
        i += 8
        self.b1 = np.array(genome[i:i+4])
        i += 4
        self.W2 = np.array(genome[i:i+8]).reshape((4, 2))
        i += 8
        self.b2 = np.array(genome[i:i+2])

    def forward(self, inputs):
        h = np.tanh(np.dot(inputs, self.W1) + self.b1)
        output = np.tanh(np.dot(h, self.W2) + self.b2)
        return output


class ANNController:
    def __init__(self):
        self.supervisor = Supervisor()
        self.robot = self.supervisor.getSelf()
        self.robot_node = self.supervisor.getFromDef("ROBOT")
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

        self.timestep = int(self.supervisor.getBasicTimeStep()*5)

        self.left_motor = self.supervisor.getDevice('motor.left')
        self.right_motor = self.supervisor.getDevice('motor.right')

        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.ground_sensors = [self.supervisor.getDevice(f'prox.ground.{i}') for i in range(2)]
        for s in self.ground_sensors:
            s.enable(self.timestep)

        self.EVALUATION_TIME = 300  # segundos
        self.collision = False
        
        # Variáveis para estratégia de exploração
        self.exploration_state = "SPIRAL"
        self.last_direction_change = 0
        self.spiral_duration = 0
        self.straight_duration = 0
        self.exploration_count = 0
        self.last_positions = []
        self.stuck_threshold = 5  # número de verificações para determinar se está preso
        self.stuck_distance_threshold = 0.05  # distância mínima para considerar movimento

    def reset_position(self):
        self.robot_node.getField('rotation').setSFRotation([0, 0, 1, np.random.uniform(0, 2*np.pi)])
        self.robot_node.getField('translation').setSFVec3f([0, 0, 0])
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        
        # Resetar variáveis de exploração
        self.exploration_state = "SPIRAL"
        self.last_direction_change = 0
        self.spiral_duration = 0
        self.straight_duration = 0
        self.exploration_count = 0
        self.last_positions = []

    def normalize(self, value):
        return (value / 1023 - 0.6) / 0.2
    
    def is_stuck(self):
        """Verifica se o robô está preso no mesmo lugar"""
        current_position = self.translation_field.getSFVec3f()
        
        if len(self.last_positions) >= self.stuck_threshold:
            # Calcular distância média percorrida nas últimas posições
            total_distance = 0
            for i in range(1, len(self.last_positions)):
                p1 = self.last_positions[i-1]
                p2 = self.last_positions[i]
                distance = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                total_distance += distance
            
            avg_distance = total_distance / (len(self.last_positions) - 1)
            
            # Se a distância média for muito pequena, está preso
            if avg_distance < self.stuck_distance_threshold:
                print(f"🔄 Robô preso! Distância média: {avg_distance:.4f}")
                self.last_positions = []  # Resetar as posições
                return True
                
            # Remover a posição mais antiga
            self.last_positions.pop(0)
            
        # Adicionar posição atual à lista
        self.last_positions.append(current_position)
        return False

    def explore(self):
        """Implementa uma estratégia de exploração para encontrar a linha"""
        current_time = self.supervisor.getTime()
        
        # Verificar se está preso em movimento circular
        if self.is_stuck():
            print("⚠️ Detectado movimento circular, mudando estratégia!")
            self.exploration_state = "STRAIGHT"
            self.straight_duration = 0
            
        # Alternar entre diferentes padrões de exploração
        if self.exploration_state == "SPIRAL":
            # Espiral/círculo grande para explorar área
            self.spiral_duration += self.timestep / 1000.0
            
            # Velocidade base e diferença de velocidade para criar um movimento em espiral
            base_speed = 3.0
            spiral_factor = min(1.5, 0.5 + self.spiral_duration / 10.0)  # Aumenta gradualmente
            
            self.left_motor.setVelocity(base_speed)
            self.right_motor.setVelocity(base_speed - spiral_factor)
            
            print(f"🌀 Explorando em espiral: Duração={self.spiral_duration:.1f}s")
            
            # Mudar para movimento reto após um certo tempo
            if self.spiral_duration > 6.0:
                self.exploration_state = "STRAIGHT"
                self.straight_duration = 0
                self.exploration_count += 1
                print("↗️ Mudando para exploração em linha reta")
                
        elif self.exploration_state == "STRAIGHT":
            # Movimento reto em uma direção aleatória
            self.straight_duration += self.timestep / 1000.0
            
            # Velocidade base para movimento reto
            straight_speed = 4.0
            
            self.left_motor.setVelocity(straight_speed)
            self.right_motor.setVelocity(straight_speed)
            
            print(f"➡️ Explorando em linha reta: Duração={self.straight_duration:.1f}s")
            
            # Mudar de direção após um período reto
            if self.straight_duration > 5.0:
                # Girar para uma nova direção aleatória
                turn_direction = np.random.choice(["LEFT", "RIGHT"])
                turn_duration = np.random.uniform(1.0, 2.0)
                
                print(f"🔄 Girando para {'esquerda' if turn_direction == 'LEFT' else 'direita'}")
                
                # Girar para uma nova direção
                start_turn_time = current_time
                while (current_time - start_turn_time) < turn_duration:
                    if turn_direction == "LEFT":
                        self.left_motor.setVelocity(0.5)
                        self.right_motor.setVelocity(3.0)
                    else:
                        self.left_motor.setVelocity(3.0)
                        self.right_motor.setVelocity(0.5)
                    
                    self.supervisor.step(self.timestep)
                    current_time = self.supervisor.getTime()
                
                # Voltar à espiral ou continuar reto com base no contador
                if self.exploration_count >= 3:
                    # Após algumas alternâncias, tentar outro padrão
                    pattern = np.random.choice(["ZIGZAG", "SPIRAL"])
                    self.exploration_state = pattern
                    self.exploration_count = 0
                    print(f"🔄 Mudando para padrão de exploração: {pattern}")
                else:
                    # Continuar reto por mais um ciclo
                    self.straight_duration = 0
                
        elif self.exploration_state == "ZIGZAG":
            # Implementação de movimento em zigzag
            zigzag_duration = 3.0  # Duração de cada segmento do zigzag
            
            # Determinar a fase do zigzag
            phase = int((current_time - self.last_direction_change) / zigzag_duration) % 2
            
            if phase == 0:
                # Diagonalmente para direita
                self.left_motor.setVelocity(4.0)
                self.right_motor.setVelocity(2.0)
                print("↗️ Zigzag: diagonal direita")
            else:
                # Diagonalmente para esquerda
                self.left_motor.setVelocity(2.0)
                self.right_motor.setVelocity(4.0)
                print("↖️ Zigzag: diagonal esquerda")
            
            # Verificar se é hora de mudar de direção
            if (current_time - self.last_direction_change) >= zigzag_duration:
                self.last_direction_change = current_time
            
            # Mudar de padrão após alguns ciclos
            if (current_time - self.last_direction_change) > zigzag_duration * 6:
                self.exploration_state = "SPIRAL"
                self.spiral_duration = 0
                print("🌀 Voltando para padrão de espiral")

    def normalize(self, value):
    # Nova função de normalização mais robusta
    # Ajuste estes valores com base nas leituras reais do seu sensor
    # Branco geralmente tem valor alto (≈1000) e preto valor mais baixo (≈500)
        white_value = 1000  # Valor aproximado para superfície branca
        black_value = 500   # Valor aproximado para linha preta
    
    # Normaliza para 0 (branco) a 1 (preto)
        normalized = max(0.0, min(1.0, (white_value - value) / (white_value - black_value)))
        return normalized

    def line_detected(self):
        """Verifica se algum sensor detectou a linha preta"""
        left_sensor = self.normalize(self.ground_sensors[0].getValue())
        right_sensor = self.normalize(self.ground_sensors[1].getValue())
    
        threshold = 0.65 # Valor mais conservador
    
        if left_sensor > threshold or right_sensor > threshold:
            print(f"📊 Valores dos sensores: L={left_sensor:.2f}, R={right_sensor:.2f} (Limiar={threshold})")
    
        return left_sensor > threshold or right_sensor > threshold

    def runStep(self, ann):
        left_raw = self.ground_sensors[0].getValue()
        right_raw = self.ground_sensors[1].getValue()

        left_sensor = self.normalize(left_raw)
        right_sensor = self.normalize(right_raw)

        print(f"Raw: L={left_raw:.1f}, R={right_raw:.1f} | Norm: L={left_sensor:.2f}, R={right_sensor:.2f}")

        # Obter posição atual para debug
        position = self.translation_field.getSFVec3f()
        print(f"Posição: X={position[0]:.2f}, Y={position[1]:.2f}, Z={position[2]:.2f}")

        inputs = np.array([left_sensor, right_sensor])
        
        # Verificar se detectou a linha
        if self.line_detected():

            motor_speeds = ann.forward(inputs) * 6  # Reduzido de 9 para 6 para maior controle
            
            # Ajustar velocidades para melhor navegação
            base_speed = 3.0
            left_speed = base_speed + motor_speeds[0] * 2
            right_speed = base_speed + motor_speeds[1] * 2
            
            # Garantir limites de velocidade razoáveis
            left_speed = max(0, min(6, left_speed))
            right_speed = max(0, min(6, right_speed))
            
            self.left_motor.setVelocity(left_speed)
            self.right_motor.setVelocity(right_speed)
            
            # Resetar contadores de exploração quando encontra a linha
            self.exploration_state = "SPIRAL"
            self.spiral_duration = 0
            self.straight_duration = 0
        else:
            # Nenhuma linha detectada - explorar

            self.explore()

        self.supervisor.step(self.timestep)

    def run(self, genome):
        self.reset_position()
        ann = SimpleANN(genome)
        start_time = self.supervisor.getTime()

        while self.supervisor.getTime() - start_time < self.EVALUATION_TIME:
            self.runStep(ann)


# Exemplo de teste com genoma predefinido para melhor desempenho
def main():
    # Genoma com valores predefinidos para melhor comportamento de seguimento de linha
    genome = [
        # W1 (2x4) - Pesos da camada de entrada para escondida
        1.0, -0.8, 0.5, -0.5,  # Sensor esquerdo
        -0.8, 1.0, -0.5, 0.5,  # Sensor direito
        
        # b1 (4) - Bias da camada escondida
        0.1, -0.1, 0.2, -0.2,
        
        # W2 (4x2) - Pesos da camada escondida para saída
        0.8, -0.5,  # Neurônio 1 -> Motores
        -0.5, 0.8,  # Neurônio 2 -> Motores
        0.3, -0.3,  # Neurônio 3 -> Motores
        -0.3, 0.3,  # Neurônio 4 -> Motores
        
        # b2 (2) - Bias da camada de saída
        0.1, 0.1  # Pequeno bias para manter movimento para frente
    ]
    
    # Descomente para usar genoma aleatório em vez do predefinido
    # genome = np.random.uniform(-1, 1, 22)
    
    controller = ANNController()
    controller.run(genome)

if __name__ == "__main__":
    main()