import sqlalchemy as db

from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()


class Station(Base):
    __tablename__ = 'station'
    id = Column(Integer, primary_key=True, autoincrement=False)
    name = Column(String(20), nullable=False, unique=True)
    type = Column(String(8), nullable=False)
    lat = Column(Float)
    lon = Column(Float)
    minutes = relationship("Minute", back_populates="station")

    def __repr__(self):
        return "Station(name='%s', type='%s')" % (self.name, self.type)


class Minute(Base):
    __tablename__ = 'minute'
    id = Column(Integer, primary_key=True, autoincrement=True)
    station_id = Column(Integer, ForeignKey('station.id'), nullable=False)
    time = Column(Integer, nullable=False)
    temp = Column(Float)
    wind_speed = Column(Float)
    wind_dir = Column(Integer)
    precip = Column(Integer)
    rvr = Column(Integer)
    mor = Column(Integer)
    station = relationship("Station", back_populates="minutes")

    # Add index on time
    __table_args__ = (db.Index('time_idx', 'time'),)

    # Unique constraint on (station_id, time)
    __table_args__ = (db.UniqueConstraint('station_id', 'time', name='_station_time_uc'),)

    def __repr__(self):
        return "Minute(station_id='%s', time='%s')" % (self.station_id, self.time)


engine = create_engine('mariadb+pymysql://lics:aelics@143.248.69.104:3306/weather')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

if __name__ == '__main__':
    from data.atfm.db.flight import Airport
    from data.atfm.db.flight import Session as FlightSession

    flight_session = FlightSession()
    incheon = flight_session.query(Airport).filter(Airport.code_icao == 'RKSI').first()
    print(incheon)

    weather_session = Session()
    station = Station(name='rksi', type='amos', lat=incheon.lat, lon=incheon.lon)
    weather_session.add(station)
    weather_session.commit()
    weather_session.close()
