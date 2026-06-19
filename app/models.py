from uuid import UUID

from pydantic import UUID4, BaseModel, StrictStr


class CustomBaseModel(BaseModel):
    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        for key, value in data.items():
            if isinstance(value, UUID):
                data[key] = str(value)
        return data


class GetMessageRequestModel(CustomBaseModel):
    dialog_id: UUID4
    last_msg_text: StrictStr
    last_message_id: UUID4 | None


class GetMessageResponseModel(CustomBaseModel):
    new_msg_text: StrictStr
    dialog_id: UUID4


class IncomingMessage(BaseModel):
    text: StrictStr
    dialog_id: UUID4
    id: UUID4
    participant_index: int


class Prediction(BaseModel):
    id: UUID4
    message_id: UUID4
    dialog_id: UUID4
    participant_index: int
    is_bot_probability: float
