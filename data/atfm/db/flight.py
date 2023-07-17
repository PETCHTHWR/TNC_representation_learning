# # Use default schema atfm_new
# USE atfm_new;
#
# CREATE TABLE IF NOT EXISTS `runway` (
#   `id` int NOT NULL AUTO_INCREMENT PRIMARY KEY,
#   `code` varchar(6) NOT NULL,
#   `start_lat` double,
#   `start_lon` double,
#   `end_lat` double,
#   `end_lon` double
# );
#
# CREATE TABLE IF NOT EXISTS `sector` (
#   `id` int NOT NULL AUTO_INCREMENT PRIMARY KEY,
#   `code` varchar(6) NOT NULL
# );
#
# CREATE TABLE IF NOT EXISTS `cluster` (
#   `id` int NOT NULL AUTO_INCREMENT PRIMARY KEY,
#   `sector_id` int NOT NULL,
#   `runway_id` int NOT NULL,
#   FOREIGN KEY (`sector_id`) REFERENCES `sector`(`id`),
#   FOREIGN KEY (`runway_id`) REFERENCES `runway`(`id`)
# );
#
# CREATE TABLE `carrier` (
#   `id` int NOT NULL AUTO_INCREMENT PRIMARY KEY,
#   `prefix_iata` varchar(2),
#   `prefix_icao` varchar(3),
#   `name_korean` varchar(255),
#   `name_english` varchar(255)
# );
#
# # Generate index for name_Korean
# CREATE INDEX IF NOT EXISTS `name_korean` ON `carrier` (`name_korean`(255));
#
# CREATE TABLE IF NOT EXISTS `airport` (
#   `id` int NOT NULL AUTO_INCREMENT PRIMARY KEY,
#   `code_iata` varchar(3),
#   `code_icao` varchar(4),
#   `name_korean` varchar(255),
#   `name_english` varchar(255)
# );
#
# # Generate index for code_iata and code_icao
# CREATE INDEX IF NOT EXISTS `code_iata` ON `airport` (`code_iata`(3));
# CREATE INDEX IF NOT EXISTS `code_icao` ON `airport` (`code_icao`(4));
#
# CREATE TABLE IF NOT EXISTS `route` (
#   `id` int NOT NULL AUTO_INCREMENT PRIMARY KEY,
#   `dep_airport_id` int NOT NULL,
#   `dest_airport_id` int NOT NULL,
#   FOREIGN KEY (`dest_airport_id`) REFERENCES `airport`(`id`),
#   FOREIGN KEY (`dep_airport_id`) REFERENCES `airport`(`id`)
# );
#
# CREATE TABLE IF NOT EXISTS `callsign` (
#   `id` int NOT NULL AUTO_INCREMENT PRIMARY KEY,
#   `carrier_id` int NOT NULL,
#   `route_id` int NOT NULL,
#   `code_iata` varchar(8),
#   `code_icao` varchar(8),
#   FOREIGN KEY (`carrier_id`) REFERENCES `carrier`(`id`),
#   FOREIGN KEY (`route_id`) REFERENCES `route`(`id`)
# );
#
# # Create index for code_iata and code_icao
# CREATE INDEX IF NOT EXISTS `code_iata` ON `callsign` (`code_iata`(8));
# CREATE INDEX IF NOT EXISTS `code_icao` ON `callsign` (`code_icao`(8));
#
# CREATE TABLE IF NOT EXISTS `flight` (
#   `id` int NOT NULL AUTO_INCREMENT PRIMARY KEY,
#   `arrival` bool NOT NULL,
#   `callsign_id` int NOT NULL,
#   `date` varchar(8) NOT NULL,
#   `sched_time` varchar(4),
#   `pred_time` varchar(4),
#   `act_time` varchar(4),
#   `runway_id` int,
#   `sector_id` int,
#   `cluster_id` int,
#   FOREIGN KEY (`runway_id`) REFERENCES `runway`(`id`),
#   FOREIGN KEY (`sector_id`) REFERENCES `sector`(`id`),
#   FOREIGN KEY (`cluster_id`) REFERENCES `cluster`(`id`),
#   FOREIGN KEY (`callsign_id`) REFERENCES `callsign`(`id`)
# );
#
# # Create index for date
# CREATE INDEX IF NOT EXISTS `date` ON `flight` (`date`(8));
#
# CREATE TABLE IF NOT EXISTS `trajectory` (
#   `id` int NOT NULL AUTO_INCREMENT PRIMARY KEY,
#   `flight_id` int NOT NULL,
#   `time` double NOT NULL,
#   `lat` double,
#   `lon` double,
#   `heading` double,
#   `vertrate` double,
#   `geoaltitude` double,
#   `baroaltitude` double,
#   FOREIGN KEY (`flight_id`) REFERENCES `flight`(`id`)
# );

import sqlalchemy as db

from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

import pandas as pd

Base = declarative_base()


class Base(Base):
    __abstract__ = True

    @staticmethod
    def to_pandas(obj):
        # If obj is a list of objects, convert it to pandas dataframe with respective dtype
        if isinstance(obj, list):
        # Convert all columns to pandas dataframe except _sa_instance_state
            df = pd.DataFrame([vars(i) for i in obj])
        # If obj is a single object, convert it to pandas dataframe
        else:
            df = pd.DataFrame([vars(obj)])
        if len(df) == 0:
            return None
        return df.drop(['_sa_instance_state'], axis=1).set_index('id')

    def __repr__(self):
        return "<{}({})>".format(self.__class__.__name__, self.id)


class Runway(Base):
    __tablename__ = 'runway'
    id = Column(Integer, primary_key=True)
    code = Column(String(6), nullable=False, unique=True)
    start_lat = Column(Float)
    start_lon = Column(Float)
    end_lat = Column(Float)
    end_lon = Column(Float)
    clusters = relationship("Cluster", back_populates="runway", cascade="all, delete-orphan")
    flights = relationship("Flight", back_populates="runway")

    # Add 'add' function that check if the runway is already in the database and return newly added runway id
    def add(self, session):
        runway = session.query(Runway).filter_by(code=self.code).first()
        if runway is None:
            session.add(self)
            session.flush()
            return self.id
        else:
            return runway.id

    # Add 'get' function that return the runway if the runway is already in the database
    def get(self, session):
        runway = session.query(Runway).filter_by(code=self.code).first()
        if runway is None:
            return None
        else:
            return runway

    # Add 'update' function that update the runway with given data for each column
    def update(self, session, data):
        runway = session.query(Runway).filter_by(code=self.code).first()
        if runway is None:
            return None
        else:
            for key, value in data.items():
                setattr(runway, key, value)
            return runway

    def __repr__(self):
        return f"Runway(id={self.id}, code={self.code}, start_lat={self.start_lat}, start_lon={self.start_lon}, end_lat={self.end_lat}, end_lon={self.end_lon})"


class Sector(Base):
    __tablename__ = 'sector'
    id = Column(Integer, primary_key=True)
    code = Column(String(6), nullable=False, unique=True)
    clusters = relationship("Cluster", back_populates="sector", cascade="all, delete-orphan")
    flights = relationship("Flight", back_populates="sector")

    # Add 'add' function that check if the sector is already in the database and return newly added sector id
    def add(self, session):
        sector = session.query(Sector).filter_by(code=self.code).first()
        if sector is None:
            session.add(self)
            session.flush()
            return self.id
        else:
            return sector.id

    # Add 'get' function that return the sector if the sector is already in the database
    def get(self, session):
        sector = session.query(Sector).filter_by(code=self.code).first()
        if sector is None:
            return None
        else:
            return sector

    # Add 'update' function that update the sector with given data for each column
    def update(self, session, data):
        sector = session.query(Sector).filter_by(code=self.code).first()
        if sector is None:
            return None
        else:
            for key, value in data.items():
                setattr(sector, key, value)
            return sector

    def __repr__(self):
        return f"Sector(id={self.id}, code={self.code})"


class Cluster(Base):
    __tablename__ = 'cluster'
    id = Column(Integer, primary_key=True)
    sector_id = Column(Integer, ForeignKey('sector.id'), nullable=False)
    runway_id = Column(Integer, ForeignKey('runway.id'), nullable=False)
    sector = relationship("Sector", back_populates="clusters")
    runway = relationship("Runway", back_populates="clusters")
    flights = relationship("Flight", back_populates="cluster")
    # Unique constraint for sector_id and runway_id
    __table_args__ = (db.UniqueConstraint('sector_id', 'runway_id', name='_sector_runway_uc'),)

    # Add 'add' function that check if the cluster is already in the database and return newly added cluster id
    def add(self, session):
        cluster = session.query(Cluster).filter_by(sector_id=self.sector_id, runway_id=self.runway_id).first()
        if cluster is None:
            session.add(self)
            session.flush()
            return self.id
        else:
            return cluster.id

    # Add 'get' function that return the cluster if the cluster is already in the database
    def get(self, session):
        cluster = session.query(Cluster).filter_by(sector_id=self.sector_id, runway_id=self.runway_id).first()
        if cluster is None:
            return None
        else:
            return cluster

    # Add 'update' function that update the cluster with given data for each column
    def update(self, session, data):
        cluster = session.query(Cluster).filter_by(sector_id=self.sector_id, runway_id=self.runway_id).first()
        if cluster is None:
            return None
        else:
            for key, value in data.items():
                setattr(cluster, key, value)
            return cluster

    def __repr__(self):
        return f"Cluster(id={self.id}, sector_id={self.sector_id}, runway_id={self.runway_id})"


class Carrier(Base):
    __tablename__ = 'carrier'
    id = Column(Integer, primary_key=True)
    prefix_iata = Column(String(2), nullable=True, unique=True)
    prefix_icao = Column(String(3), nullable=True, unique=True)
    name_korean = Column(String(255))
    name_english = Column(String(255))
    callsigns = relationship("Callsign", back_populates="carrier", cascade="all, delete-orphan")

    # Add 'add' function that check if the carrier is already in the database and return newly added carrier id
    def add(self, session):
        if self.prefix_iata is not None and self.prefix_icao is not None:
            carrier = session.query(Carrier).filter_by(prefix_iata=self.prefix_iata,
                                                       prefix_icao=self.prefix_icao).first()
        elif self.prefix_iata is not None:
            carrier = session.query(Carrier).filter_by(prefix_iata=self.prefix_iata).first()
        elif self.prefix_icao is not None:
            carrier = session.query(Carrier).filter_by(prefix_icao=self.prefix_icao).first()
        else:
            raise Exception("Both prefix_iata and prefix_icao are None")
        if carrier is None:
            session.add(self)
            session.flush()
            return self.id
        else:
            return carrier.id

    # Add 'get' function that return the carrier if the carrier is already in the database
    def get(self, session):
        if self.prefix_iata is not None and self.prefix_icao is not None:
            carrier = session.query(Carrier).filter_by(prefix_iata=self.prefix_iata,
                                                       prefix_icao=self.prefix_icao).first()
        elif self.prefix_iata is not None:
            carrier = session.query(Carrier).filter_by(prefix_iata=self.prefix_iata).first()
        elif self.prefix_icao is not None:
            carrier = session.query(Carrier).filter_by(prefix_icao=self.prefix_icao).first()
        else:
            raise Exception("Both prefix_iata and prefix_icao are None")
        if carrier is None:
            return None
        else:
            return carrier

    # Add 'update' function that update the carrier with given data for each column
    def update(self, session, data):
        if self.prefix_iata is not None and self.prefix_icao is not None:
            carrier = session.query(Carrier).filter_by(prefix_iata=self.prefix_iata,
                                                       prefix_icao=self.prefix_icao).first()
        elif self.prefix_iata is not None:
            carrier = session.query(Carrier).filter_by(prefix_iata=self.prefix_iata).first()
        elif self.prefix_icao is not None:
            carrier = session.query(Carrier).filter_by(prefix_icao=self.prefix_icao).first()
        else:
            raise Exception("Both prefix_iata and prefix_icao are None")
        if carrier is None:
            return None
        else:
            for key, value in data.items():
                setattr(carrier, key, value)
            return carrier

    def __repr__(self):
        return f"Carrier(id={self.id}, prefix_iata={self.prefix_iata}, prefix_icao={self.prefix_icao}, name_korean={self.name_korean}, name_english={self.name_english})"


class Airport(Base):
    __tablename__ = 'airport'
    id = Column(Integer, primary_key=True)
    code_iata = Column(String(3), nullable=True, unique=True)
    code_icao = Column(String(4), nullable=True, unique=True)
    name_korean = Column(String(255))
    name_english = Column(String(255))
    lat = Column(Float)
    lon = Column(Float)
    altitude = Column(Float)

    # Add 'add' function that check if the airport is already in the database and return newly added airport id
    def add(self, session):
        if self.code_iata is not None and self.code_icao is not None:
            airport = session.query(Airport).filter_by(code_iata=self.code_iata, code_icao=self.code_icao).first()
        elif self.code_iata is not None:
            airport = session.query(Airport).filter_by(code_iata=self.code_iata).first()
        elif self.code_icao is not None:
            airport = session.query(Airport).filter_by(code_icao=self.code_icao).first()
        else:
            raise Exception("Both code_iata and code_icao are None")
        if airport is None:
            session.add(self)
            session.flush()
            return self.id
        else:
            return airport.id

    # Add 'get' function that return the airport if the airport is already in the database
    def get(self, session):
        if self.code_iata is not None and self.code_icao is not None:
            airport = session.query(Airport).filter_by(code_iata=self.code_iata, code_icao=self.code_icao).first()
        elif self.code_iata is not None:
            airport = session.query(Airport).filter_by(code_iata=self.code_iata).first()
        elif self.code_icao is not None:
            airport = session.query(Airport).filter_by(code_icao=self.code_icao).first()
        else:
            raise Exception("Both code_iata and code_icao are None")
        if airport is None:
            return None
        else:
            return airport

    # Add 'update' function that update the airport with given data for each column
    def update(self, session, data):
        if self.code_iata is not None and self.code_icao is not None:
            airport = session.query(Airport).filter_by(code_iata=self.code_iata, code_icao=self.code_icao).first()
        elif self.code_iata is not None:
            airport = session.query(Airport).filter_by(code_iata=self.code_iata).first()
        elif self.code_icao is not None:
            airport = session.query(Airport).filter_by(code_icao=self.code_icao).first()
        else:
            raise Exception("Both code_iata and code_icao are None")
        if airport is None:
            return None
        else:
            for key, value in data.items():
                setattr(airport, key, value)
            return airport

    def __repr__(self):
        return f"Airport(id={self.id}, code_iata={self.code_iata}, code_icao={self.code_icao}, name_korean={self.name_korean}, name_english={self.name_english})"


class Route(Base):
    __tablename__ = 'route'
    id = Column(Integer, primary_key=True)
    dep_airport_id = Column(Integer, ForeignKey('airport.id'), nullable=False)
    dest_airport_id = Column(Integer, ForeignKey('airport.id'), nullable=False)
    callsigns = relationship("Callsign", back_populates="route", cascade="all, delete-orphan")
    # Generate cascaded flights relationship through callsign.id -> flights
    # Unique constraint for dep_airport_id and dest_airport_id
    __table_args__ = (db.UniqueConstraint('dep_airport_id', 'dest_airport_id', name='_dep_dest_uc'),)

    # Add 'add' function that check if the route is already in the database and return newly added route id
    def add(self, session):
        route = session.query(Route).filter_by(dep_airport_id=self.dep_airport_id,
                                               dest_airport_id=self.dest_airport_id).first()
        if route is None:
            session.add(self)
            session.flush()
            return self.id
        else:
            return route.id

    # Add 'get' function that return the route if the route is already in the database
    def get(self, session):
        route = session.query(Route).filter_by(dep_airport_id=self.dep_airport_id,
                                               dest_airport_id=self.dest_airport_id).first()
        if route is None:
            return None
        else:
            return route

    # Add 'update' function that update the route with given data for each column
    def update(self, session, data):
        route = session.query(Route).filter_by(dep_airport_id=self.dep_airport_id,
                                               dest_airport_id=self.dest_airport_id).first()
        if route is None:
            return None
        else:
            for key, value in data.items():
                setattr(route, key, value)
            return route

    def __repr__(self):
        return f"Route(id={self.id}, dep_airport_id={self.dep_airport_id}, dest_airport_id={self.dest_airport_id})"


class Callsign(Base):
    __tablename__ = 'callsign'
    id = Column(Integer, primary_key=True)
    carrier_id = Column(Integer, ForeignKey('carrier.id'), nullable=False)
    route_id = Column(Integer, ForeignKey('route.id'), nullable=False)
    code_iata = Column(String(8), nullable=True, unique=True)
    code_icao = Column(String(8), nullable=True, unique=True)
    # type
    # 0: passenger
    # 1: cargo
    # 2: other
    type = Column(Integer, nullable=False)

    carrier = relationship("Carrier", back_populates="callsigns")
    route = relationship("Route", back_populates="callsigns")
    flights = relationship("Flight", back_populates="callsign", cascade="all, delete-orphan")

    # Add 'add' function that check if the callsign is already in the database and return newly added callsign id
    def add(self, session):
        if self.code_iata is not None and self.code_icao is not None:
            callsign = session.query(Callsign).filter_by(code_iata=self.code_iata, code_icao=self.code_icao).first()
        elif self.code_iata is not None:
            callsign = session.query(Callsign).filter_by(code_iata=self.code_iata).first()
        elif self.code_icao is not None:
            callsign = session.query(Callsign).filter_by(code_icao=self.code_icao).first()
        else:
            raise Exception("Both code_iata and code_icao are None")
        if callsign is None:
            session.add(self)
            session.flush()
            return self.id
        else:
            return callsign.id

    # Add 'get' function that return the callsign if the callsign is already in the database
    def get(self, session):
        if self.code_iata is not None and self.code_icao is not None:
            callsign = session.query(Callsign).filter_by(code_iata=self.code_iata, code_icao=self.code_icao).first()
        elif self.code_iata is not None:
            callsign = session.query(Callsign).filter_by(code_iata=self.code_iata).first()
        elif self.code_icao is not None:
            callsign = session.query(Callsign).filter_by(code_icao=self.code_icao).first()
        else:
            raise Exception("Both code_iata and code_icao are None")
        if callsign is None:
            return None
        else:
            return callsign

    # Add 'update' function that update the callsign with given data for each column
    def update(self, session, data):
        if self.code_iata is not None and self.code_icao is not None:
            callsign = session.query(Callsign).filter_by(code_iata=self.code_iata, code_icao=self.code_icao).first()
        elif self.code_iata is not None:
            callsign = session.query(Callsign).filter_by(code_iata=self.code_iata).first()
        elif self.code_icao is not None:
            callsign = session.query(Callsign).filter_by(code_icao=self.code_icao).first()
        else:
            raise Exception("Both code_iata and code_icao are None")
        if callsign is None:
            return None
        else:
            for key, value in data.items():
                setattr(callsign, key, value)
            return callsign

    def __repr__(self):
        return f"Callsign(id={self.id}, carrier_id={self.carrier_id}, route_id={self.route_id}, code_iata={self.code_iata}, code_icao={self.code_icao})"


class Flight(Base):
    __tablename__ = 'flight'
    id = Column(Integer, primary_key=True)
    arrival = Column(Boolean, nullable=False)
    callsign_id = Column(Integer, ForeignKey('callsign.id'), nullable=False)
    date = Column(String(8), nullable=False)
    sched_time = Column(String(4))
    pred_time = Column(String(4))
    act_time = Column(String(4))
    runway_id = Column(Integer, ForeignKey('runway.id'))
    sector_id = Column(Integer, ForeignKey('sector.id'))
    cluster_id = Column(Integer, ForeignKey('cluster.id'))
    # Unique constraint for callsign_id, date, sched_time
    __table_args__ = (db.UniqueConstraint('callsign_id', 'date', 'sched_time', name='_callsign_date_sched_uc'),)

    callsign = relationship("Callsign", back_populates="flights")
    # Generate route relationship using secondary join on callsign table
    # route = relationship("Route", secondary="callsign", back_populates="flights")
    runway = relationship("Runway", back_populates="flights")
    sector = relationship("Sector", back_populates="flights")
    cluster = relationship("Cluster", back_populates="flights")
    trajectories = relationship("Trajectory", back_populates="flight", cascade="all, delete-orphan")

    # Add 'add' function that check if the flight is already in the database and return newly added flight id
    def add(self, session):
        flight = session.query(Flight).filter_by(callsign_id=self.callsign_id, date=self.date,
                                                 sched_time=self.sched_time).first()
        if flight is None:
            session.add(self)
            session.flush()
            return self.id
        else:
            return flight.id

    # Add 'get' function that return the flight if the flight is already in the database
    def get(self, session):
        flight = session.query(Flight).filter_by(callsign_id=self.callsign_id, date=self.date,
                                                 sched_time=self.sched_time).first()
        if flight is None:
            return None
        else:
            return flight

    # Add 'update' function that update the flight with given data for each column
    def update(self, session, data):
        flight = session.query(Flight).filter_by(callsign_id=self.callsign_id, date=self.date,
                                                 sched_time=self.sched_time).first()
        if flight is None:
            return None
        else:
            for key, value in data.items():
                setattr(flight, key, value)
            return flight

    def __repr__(self):
        return f"Flight(id={self.id}, arrival={self.arrival}, callsign_id={self.callsign_id}, date={self.date}, sched_time={self.sched_time}, pred_time={self.pred_time}, act_time={self.act_time}, runway_id={self.runway_id}, sector_id={self.sector_id}, cluster_id={self.cluster_id})"


class Trajectory(Base):
    __tablename__ = 'trajectory'
    id = Column(Integer, primary_key=True)
    flight_id = Column(Integer, ForeignKey('flight.id'), nullable=False)
    time = Column(Integer, nullable=False)
    lat = Column(Float)
    lon = Column(Float)
    velocity = Column(Float)
    heading = Column(Float)
    vertrate = Column(Float)
    geoaltitude = Column(Float)
    baroaltitude = Column(Float)
    flight = relationship("Flight", back_populates="trajectories")
    # Unique constraint for flight_id and time
    __table_args__ = (db.UniqueConstraint('flight_id', 'time', name='_flight_time_uc'),)

    # Add 'add' function that check if the trajectory is already in the database and return newly added trajectory id
    def add(self, session):
        trajectory = session.query(Trajectory).filter_by(flight_id=self.flight_id, time=self.time).first()
        if trajectory is None:
            session.add(self)
            session.flush()
            return self.id
        else:
            return trajectory.id

    # Add 'get' function that return the trajectory if the trajectory is already in the database
    def get(self, session):
        trajectory = session.query(Trajectory).filter_by(flight_id=self.flight_id, time=self.time).first()
        if trajectory is None:
            return None
        else:
            return trajectory

    # Add 'update' function that update the trajectory with given data for each column
    def update(self, session, data):
        trajectory = session.query(Trajectory).filter_by(flight_id=self.flight_id, time=self.time).first()
        if trajectory is None:
            return None
        else:
            for key, value in data.items():
                setattr(trajectory, key, value)
            return trajectory

    def __repr__(self):
        return f"Trajectory(id={self.id}, flight_id={self.flight_id}, time={self.time}, lat={self.lat}, lon={self.lon}, heading={self.heading}, vertrate={self.vertrate}, geoaltitude={self.geoaltitude}, baroaltitude={self.baroaltitude})"


engine = create_engine('mariadb+pymysql://lics:aelics070@143.248.69.104:3306/atfm_new')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

if __name__ == '__main__':
    session = Session()

    carriers = session.query(Carrier).all()
    print(Carrier.to_pandas(carriers))

    # for carrier in carriers:
    #     print(carrier.callsigns)
    #     break
