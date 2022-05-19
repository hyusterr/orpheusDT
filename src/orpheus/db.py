import click
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

class DatabaseManager:
    def __init__(self):
        self.client = self._connect_db()

        self.database = None
        self.dataset = None
        self.model = None
    
    def _connect_db(self):
        """
        Connect to MongoDB server.

        Returns
        ----------
        client : Client for a MongoDB instance.
        """
        # TODO: should use username and password
        client = MongoClient(host='localhost', port=27017)

        # Check connection
        try:
            client.admin.command('ping') # The ping command is cheap and does not require auth.
        except ConnectionFailure:
            click.secho("Cannot connect to the database.", fb='red')
        
        return client
    
    def use_database(self, db_name: str):
        """
        Parameters
        ----------
        db_name : Name of database.
        """
        if db_name not in self.client.list_database_names():
            click.secho(f'Note: Database {db_name} not exists. Creating a new one ...', fg='yellow')
        
        self.database = self.client[db_name]
        self.dataset = self.database["dataset"]
        self.model = self.database["model"]

    def drop_database(self):
        """
        Drop current database (database) from the MongoDB client.
        """
        if self.database is not None:
            self.client.drop_database(self.database)
            click.secho(f'Current database dropped successfully.')
        else:
            click.secho(f'Failed: has not specified any database.', fg='red')