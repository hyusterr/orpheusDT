import os
from datetime import datetime
import click
from typing import List
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure
import bson


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
        self.name = dbname
        self.client = self._connect_db(username, password)

        # Create database
        if dbname not in self.client.list_database_names():
            click.secho(f'Note: Database {dbname} not exists. Creating a new one ...', fg='yellow')

        self.db = Database(self.client, name=dbname)
        self.data = self.db.get_collection('data')
        self.model = self.db.get_collection('model')
    
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
            username=username,
            password=password,
            authSource=self.name
        )

        # Check connection
        try:
            client.admin.command('ping') # The ping command is cheap and does not require auth.
        except ConnectionFailure:
            click.secho("Cannot connect to the database.", fb='red')
        
        return client

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

    def back_up_collection(self, collection_name):
        """
        Back up current collection with json file
        """
        crt_time = int(round(datetime.now().timestamp()))
        path = f'./backup/{self.db.name}/{collection_name}'
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            files_name = os.listdir(path)
            # check whether here are too many backup file and remove old file
            if len(files_name) >= 20:
                # sort by file name (remove file extension)
                files_name.sort(key=lambda x:int(x[:-5]))
                for i in range(10):
                    os.remove(f'{path}/{files_name[i]}')

        with open(f'{path}/{crt_time}.json', 'wb+') as f:
            current_collection = self.db[collection_name]
            
            cursor = current_collection.find({})
            for c in cursor:
                f.write(bson.BSON.encode(c))

    def restore_collection(self, collection_name):
        """
        Restore collection which is latest file
        """
        path = f'./backup/{self.db.name}/{collection_name}'
        files_name = os.listdir(path)
        # sort by file name (remove file extension)
        files_name.sort(key=lambda x:int(x[:-5]))
        file_name = files_name[-1]

        with open(f'{path}/{file_name}', 'rb') as f:
            current_collection = self.db[collection_name]
            backup_data = f.read()
            current_collection.delete_many({})
            current_collection.insert_many(bson.decode_all(backup_data))