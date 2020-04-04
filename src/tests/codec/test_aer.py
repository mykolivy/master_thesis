import pytest
import src.event_compression.codec as codec
from src.event_compression.codec import aer as aer

@pytest.fixture
def smtp_connection():
    import smtplib
    return smtplib.SMTP("smtp.gmail.com", 587, timeout=5)

def test_init():
	assert codec.codecs()