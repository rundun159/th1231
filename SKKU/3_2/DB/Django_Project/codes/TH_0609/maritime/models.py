from django.db import models
class Gov(models.Model):
    gov_id = models.CharField(max_length=20, help_text='log in ID of Government',primary_key=True)
    gov_passwd = models.CharField(max_length=20, help_text='log in password of Government')
class Comp(models.Model):   #company
    comp_name = models.CharField(max_length=20, help_text='name of Company',primary_key=True)
    comp_passwd = models.CharField(max_length=20, help_text='log in password of Company')
    comp_id = models.CharField(max_length=20, help_text='log in ID of Company')
class Mate_temp(models.Model):
    mate_name = models.CharField(max_length=20, help_text='name of mate')
    mate_comp = models.ForeignKey(Comp,on_delete=models .CASCADE)
    gender = models.CharField(max_length=10) #male / female
    age = models.IntegerField(default=0)
class Mate_stored(models.Model):
    mate_name = models.CharField(max_length=20, help_text='name of mate')
    mate_comp = models.ForeignKey(Comp,on_delete=models.CASCADE)
    gender = models.CharField(max_length=10) #male / female
    age = models.IntegerField(default=0)
class Mate_rejected(models.Model):
    mate_name = models.CharField(max_length=20, help_text='name of mate')
    mate_comp = models.ForeignKey(Comp,on_delete=models.CASCADE)
    gender = models.CharField(max_length=10) #male / female
    age = models.IntegerField(default=0)
class Eng_temp(models.Model):
    eng_name = models.CharField(max_length=20, help_text='name of Engineer')
    eng_comp = models.ForeignKey(Comp,on_delete=models.CASCADE)
    gender = models.CharField(max_length=10) #male / female
    age = models.IntegerField(default=0)
class Eng_stored(models.Model):
    eng_name = models.CharField(max_length=20, help_text='name of Engineer')
    eng_comp = models.ForeignKey(Comp,on_delete=models.CASCADE)
    gender = models.CharField(max_length=10) #male / female
    age = models.IntegerField(default=0)
class Eng_rejected(models.Model):
    eng_name = models.CharField(max_length=20, help_text='name of Engineer')
    eng_comp = models.ForeignKey(Comp,on_delete=models.CASCADE)
    gender = models.CharField(max_length=10) #male / female
    age = models.IntegerField(default=0)
class Ship_temp(models.Model):
    ship_name = models.CharField(max_length=20, help_text='name of ship',primary_key=True)
    ship_comp = models.ForeignKey(Comp,on_delete=models.CASCADE)
    age = models.IntegerField(default=0)
class Ship_stored(models.Model):
    ship_name = models.CharField(max_length=20, help_text='name of ship',primary_key=True)
    ship_comp = models.ForeignKey(Comp,on_delete=models.CASCADE)
    age = models.IntegerField(default=0)
class Ship_rejected(models.Model):
    ship_name = models.CharField(max_length=20, help_text='name of ship',primary_key=True)
    ship_comp = models.ForeignKey(Comp,on_delete=models.CASCADE)
    age = models.IntegerField(default=0)
class Port(models.Model):
    port_name = models.CharField(max_length=30, help_text='name of port',primary_key=True)
class Plan_temp(models.Model):
    plan_port1 = models.ForeignKey(Port,on_delete=models.CASCADE,related_name='depart_temp')
    plan_port2 = models.ForeignKey(Port,on_delete=models.CASCADE,related_name='arrive_temp')
    departure = models.DateField()
    arrival = models.DateField()
    plan_ship = models.ForeignKey(Ship_stored, on_delete=models.CASCADE)
    plan_comp = models.ForeignKey(Comp, on_delete=models.CASCADE)
class Plan_stored(models.Model):
    plan_port1 = models.ForeignKey(Port,on_delete=models.CASCADE,related_name='depart_stored')
    plan_port2 = models.ForeignKey(Port,on_delete=models.CASCADE,related_name='arrive_stored')
    departure = models.DateField()
    arrival = models.DateField()
    plan_ship = models.ForeignKey(Ship_stored, on_delete=models.CASCADE)
    plan_comp = models.ForeignKey(Comp, on_delete=models.CASCADE)
class Plan_rejected(models.Model):
    plan_port1 = models.ForeignKey(Port,on_delete=models.CASCADE,related_name='depart_rejected')
    plan_port2 = models.ForeignKey(Port,on_delete=models.CASCADE,related_name='arrive_rejected')
    departure = models.DateField()
    arrival = models.DateField()
    plan_ship = models.ForeignKey(Ship_stored, on_delete=models.CASCADE)
    plan_comp = models.ForeignKey(Comp, on_delete=models.CASCADE)
class Modify_Plan(models.Model):
    departure = models.DateField()
    arrival = models.DateField()

