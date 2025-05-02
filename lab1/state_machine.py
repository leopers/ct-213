import random
import math
from constants import *


class FiniteStateMachine(object):
    """
    A finite state machine.
    """

    def __init__(self, state):
        self.state = state

    def change_state(self, new_state):
        self.state = new_state

    def update(self, agent):
        self.state.check_transition(agent, self)
        self.state.execute(agent)


class State(object):
    """
    Abstract state class.
    """

    def __init__(self, state_name):
        """
        Creates a state.

        :param state_name: the name of the state.
        :type state_name: str
        """
        self.state_name = state_name

    def check_transition(self, agent, fsm):
        """
        Checks conditions and execute a state transition if needed.

        :param agent: the agent where this state is being executed on.
        :param fsm: finite state machine associated to this state.
        """
        raise NotImplementedError(
            "This method is abstract and must be implemented in derived classes"
        )

    def execute(self, agent):
        """
        Executes the state logic.

        :param agent: the agent where this state is being executed on.
        """
        raise NotImplementedError(
            "This method is abstract and must be implemented in derived classes"
        )


class MoveForwardState(State):
    def __init__(self):
        super().__init__("MoveForward")
        self.decisions = 0

    def check_transition(self, agent, state_machine):
        if agent.get_bumper_state():
            state_machine.change_state(GoBackState())
        if self.decisions > MOVE_FORWARD_TIME / SAMPLE_TIME:
            state_machine.change_state(MoveInSpiralState())
        self.decisions += 1

    def execute(self, agent):
        agent.set_velocity(FORWARD_SPEED, 0)


class MoveInSpiralState(State):
    def __init__(self):
        super().__init__("MoveInSpiral")
        self.decisions = 0

    def check_transition(self, agent, state_machine):
        if agent.get_bumper_state():
            state_machine.change_state(GoBackState())
        if self.decisions > MOVE_IN_SPIRAL_TIME / SAMPLE_TIME:
            state_machine.change_state(MoveForwardState())
        self.decisions += 1

    def execute(self, agent):
        time = SAMPLE_TIME * self.decisions
        radius = INITIAL_RADIUS_SPIRAL + SPIRAL_FACTOR * time
        angular_speed = FORWARD_SPEED / radius
        agent.set_velocity(FORWARD_SPEED, angular_speed)


class GoBackState(State):
    def __init__(self):
        super().__init__("GoBack")
        self.decisions = 0

    def check_transition(self, agent, state_machine):
        if self.decisions > GO_BACK_TIME / SAMPLE_TIME:
            state_machine.change_state(RotateState())
        self.decisions += 1

    def execute(self, agent):
        agent.set_velocity(BACKWARD_SPEED, 0)


class RotateState(State):
    def __init__(self):
        super().__init__("Rotate")
        self.decisions = 0
        self.angle = random.uniform(-math.pi, math.pi)
        self.direction = 1 if self.angle >= 0 else -1
        self.rotation_time = abs(self.angle) / ANGULAR_SPEED

    def check_transition(self, agent, state_machine):
        if self.decisions > self.rotation_time / SAMPLE_TIME:
            state_machine.change_state(MoveForwardState())
        self.decisions += 1

    def execute(self, agent):
        agent.set_velocity(0, self.direction * ANGULAR_SPEED)
