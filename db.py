import click
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

class DatabaseManager:
    def __init__(self):
        self.client = self._connect_db()

        self.project = None
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
    
    def use_project(self, prj_name: str):
        """
        Parameters
        ----------
        prj_name : Name of project.
        """
        if prj_name not in self.client.list_database_names():
            click.secho(f'Note: Project {prj_name} not exists. Creating a new one ...', fg='yellow')
        
        self.project = self.client[prj_name]
        self.dataset = self.project["dataset"]
        self.model = self.project["model"]

    def drop_project(self):
        """
        Drop current project (database) from the MongoDB client.
        """
        if self.project is not None:
            self.client.drop_database(self.project)
            click.secho(f'Current project dropped successfully.')
        else:
            click.secho(f'Failed: has not specified any project.', fg='red')