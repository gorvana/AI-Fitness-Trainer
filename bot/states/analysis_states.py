from aiogram.fsm.state import State, StatesGroup

class AnalysisStates(StatesGroup):
    waiting_for_video = State()
    processing_video = State()
    waiting_for_feedback = State()