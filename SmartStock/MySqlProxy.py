import pymysql

"""
数据库操作：1、链接;2、创建库;3、查看库;4、使用库;5、创建表;
          6、查看表; 7、查询; 8、插入;9、删除;10、修改
"""

class MySqlProxy:
    __IP = "localhost"
    __user = "root"
    __psw = "1234"
    __database = "test_table"

    def __init__(self, ip="localhost", user="root", psw="123456", database="test_table"):
        self.__IP = ip
        self.__user = user
        self.__psw = psw
        self.__database = database
        self.__db = None

    def __open(self):
        self.__db = pymysql.connect(self.__IP, self.__user, self.__psw,
                                    self.__database)
        cursor = self.__db.cursor()
        cursor.execute("use %s" % self.__database)
        cursor.close()

    def __close(self):
        self.__db.close()

    def exec(self, command):
        self.__open()
        cursor = self.__db.cursor()
        cursor.execute(command)
        data = cursor.fetchall()
        cursor.close()
        self.__close()
        return data


    def chooseDatabase(self, name):
        self.__database = name

    def createDatabase(self, name):
        self.__db = pymysql.connect(self.__IP, self.__user, self.__psw,
                                    self.__database)
        cursor = self.__db.cursor()
        cursor.execute("create database " + name)




if __name__ == "__main__":
    print("hello")
    mydatabase = MySqlProxy()
    data = mydatabase.exec("SELECT VERSION()")
    print(data)
    data = mydatabase.exec("show databases")
    print(data)
    data = mydatabase.exec("select database()")
    print(data)
    data = mydatabase.query("show create database information_schema")
    print(data)

