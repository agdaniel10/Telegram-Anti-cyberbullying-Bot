from sqlalchemy import create_engine, Column, Integer, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

MAXIMUM_BULLY_MESSAGES = 8
MAXIMUM_DAYS_BANNED = 1

class UserOffense(Base):
    __tablename__ = 'user_offenses'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    chat_id = Column(Integer, nullable=False)
    no_bullying = Column(Integer, default=0)
    last_bullying_time = Column(DateTime, nullable=True)

DATABASE_URL = "sqlite:///cyberbullying_bot.db"
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

# Define a whitelist of non-abusive mentions
WHITELIST_MENTIONS = ["@anti_abuse_and_cyberbullying_bot"]

def add_offense(user_id, chat_id, message):
    with SessionLocal() as session:
        # Check if the message mentions a whitelisted entity
        if any(mention in message for mention in WHITELIST_MENTIONS):
            return  # Don't record an offense for whitelisted mentions

        offense = session.query(UserOffense).filter_by(user_id=user_id, chat_id=chat_id).first()
        current_time = datetime.now()

        if offense:
            # Check if enough time has passed to reset offenses
            time_diff = current_time - (offense.last_bullying_time or datetime.min)
            if time_diff.total_seconds() > 14 * 3600:  # Reset after 14 hours
                offense.no_bullying = 0

            offense.no_bullying += 1  
            offense.last_bullying_time = current_time

            # Check if the user has exceeded the maximum number of offenses
            if offense.no_bullying >= MAXIMUM_BULLY_MESSAGES:
                print(f"User {user_id} in chat {chat_id} is banned for {MAXIMUM_DAYS_BANNED} days due to too many offensive messages.")

        else:
            offense = UserOffense(user_id=user_id, chat_id=chat_id, no_bullying=1, last_bullying_time=current_time)
            session.add(offense)

        session.commit()

def get_offense_count(user_id, chat_id):
    with SessionLocal() as session:
        offense = session.query(UserOffense).filter_by(user_id=user_id, chat_id=chat_id).first()
        return offense.no_bullying if offense else 0

def reset_offense(user_id, chat_id):
    with SessionLocal() as session:
        offense = session.query(UserOffense).filter_by(user_id=user_id, chat_id=chat_id).first()
        if offense:
            offense.no_bullying = 0
            offense.last_bullying_time = None
            session.commit()
