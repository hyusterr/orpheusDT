import click
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

class DatabaseManager:
    def __init__(self, dbname: str):
        """
        Parameters
        ----------
        db_name : Specify the working database.
        """
        self.client = self._connect_db()

        if dbname not in self.client.list_database_names():
            click.secho(f'Note: Database {dbname} not exists. Creating a new one ...', fg='yellow')
        
        self.database = self.client[dbname]
        self.dataset = self.database["dataset"]
        self.model = self.database["model"]
    
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

    def drop_database(self):
        """
        Drop current database from the MongoDB client.
        TODO: This is a dangerous operation and thus should require authentication.
        """
        if self.database is not None:
            self.client.drop_database(self.database)
            click.secho(f'Current database dropped successfully.')
        else:
            click.secho(f'Failed: has not specified any database.', fg='red')
