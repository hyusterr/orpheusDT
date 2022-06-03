

import click
from typing import List
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure


class DatabaseManager:
    def __init__(
            self,
            dbname: str,
            username: str = None,
            password: str = None
    ):
        """
        Parameters
        ----------
        dbname : Specify the working database.
        """
        self.dbname = dbname
        self.client = self._connect_db(username, password)

        # Create database
        if dbname not in self.client.list_database_names():
            click.secho(f'Note: Database {dbname} not exists. Creating a new database for task {dbname}', fg='yellow')

        self.db = Database(self.client, name=dbname)
        self.data_collection = self.db.get_collection('data_collection')
        self.metadata_collection = self.db.get_collection('metadata_collection')

    def _connect_db(self, username: str = None, password: str = None):
        """
        Connect to MongoDB server.
        Returns
        ----------
        client : Client for a MongoDB instance.
        """
        client = MongoClient(
            host='localhost',
            port=27017,
            # username=username,
            # password=password,
            # authSource=self.dbname,
            # authMechanism='SCRAM-SHA-256'
        )

        # Check connection
        try:
            client.admin.command('ping')  # The ping command is cheap and does not require auth.
        except ConnectionFailure:
            click.secho("Cannot connect to the database.", fb='red')

        return client

    def insert_document(self, collection, document):

        current_collection = self.db[collection]
        current_collection.insert_one(document)


    def query_document(self, collection, query_key, query_value):
        current_collection = self.db[collection]
        cursor = current_collection.find_one({query_key: query_value})

        return cursor

    def custom_query(self, collection, query_filter, query_preojeciton):
        current_collection = self.db[collection]
        cursor = current_collection.find(query_filter, query_preojeciton)

        return cursor

    def custom_aggregation(self, collection, agg_match, agg_project, agg_sort):
        current_collection = self.db[collection]

        cursor = current_collection.aggregate([agg_match, agg_project, agg_sort])

        return cursor


    def count_Document(self, collection, count_filter):
        current_collection = self.db[collection]
        num = current_collection.count_documents(count_filter)

        return num


    # user management
    def create_user(self, user: str, pwd: str, roles: List[str]):
        self.db.command('createUser', user, pwd=pwd, roles=roles)

    def grant_roles(self, user: str, roles: List[str]):
        self.db.command('grantRolesToUser', user, roles=roles)

    # database management
    def drop_database(self):
        """
        Drop current database from the MongoDB client.
        TODO: This is a dangerous operation and thus should require authentication.
        """
        if self.db is not None:
            self.client.drop_database(self.db)
            click.secho(f'Current database dropped successfully.')
        else:
            click.secho(f'Failed: has not specified any database.', fg='red')