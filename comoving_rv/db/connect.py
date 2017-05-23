from __future__ import division, print_function

# Standard library
import os

# Third-party
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base

Session = scoped_session(sessionmaker(autoflush=True, autocommit=False))
Base = declarative_base()

def db_connect(database_path, ensure_db_exists=True):
    """
    Connect to the specified database.

    Parameters
    ----------
    database : str (optional)
        Name of database
    ensure_db_exists : bool (optional)
        Ensure the database ``database`` exists.

    Returns
    -------
    engine :
        The sqlalchemy database engine.
    """

    engine = create_engine("sqlite:///{}".format(os.path.abspath(database_path)))
    Session.configure(bind=engine)
    Base.metadata.bind = engine

    if ensure_db_exists:
        Base.metadata.create_all(engine)

    return engine
