from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse, reverse_lazy
from .models import *
from .forms import *
from .models import Mate_stored, Mate_temp, Comp, Eng_stored, Eng_temp, Ship_stored, Ship_temp, Plan_stored, Plan_temp, Port
from django.views.generic import CreateView

def signup(request):
    form = SigUpForm()
    return render(request,'accounts/signup.html', {'form': form})
def comp_main_page(request):
    return render(request,'accounts/comp_main.html')
def maritime_main(request):
    return render(request,'accounts/maritime_main.html')
def gov_main_page(request):
    return render(request,'gov/gov_main_page.html')
def signup_comp(request):
    form = SigUpForm(request.POST)
    if form.is_valid():
        new_item = form.save()
        new_item.save()
        data = form.cleaned_data
    else:
        print("Erro TH 2")
    return HttpResponseRedirect(reverse('maritime:comp_main',args=(data['comp_name'],)))


def login(request):
    form = Comp_LogInForm()
    return render(request,'accounts/login.html', {'form': form})

def connect_to_main(request):
    form = Comp_LogInForm(request.POST)
    if form.is_valid():
        data = form.cleaned_data
    return HttpResponseRedirect(reverse('maritime:comp_main',args=(data['comp_id'],)))

def comp_main(request,comp_name):
    return render(request,'comp/comp_main.html',{'comp_name': comp_name})

def comp_enroll_select(request, comp_name):
    return render(request,'comp/comp_enroll_select.html', {'comp_name': comp_name })

def comp_enroll_mate(request, comp_name):
    if request.method == 'POST':
        form = MateForm(request.POST)
        if form.is_valid():
            new_item = form.save()
        new_item.save()
        return HttpResponseRedirect(reverse('maritime:comp_main', args=(comp_name,)))
    mate_form = MateForm()
    return render(request,'comp/comp_enroll_mate.html', {'form': mate_form, 'comp_name': comp_name})

def comp_enroll_eng(request, comp_name):
    if request.method == 'POST':
        form = EngForm(request.POST)
        if form.is_valid():
            new_item = form.save()
        new_item.save()
        return HttpResponseRedirect(reverse('maritime:comp_main', args=(comp_name,)))
    form = EngForm()
    return render(request,'comp/comp_enroll_eng.html', {'form': form, 'comp_name': comp_name})

def comp_enroll_ship(request, comp_name):
    if request.method == 'POST':
        form = ShipForm(request.POST)
        if form.is_valid():
            new_item = form.save()
        new_item.save()
        return HttpResponseRedirect(reverse('maritime:comp_main', args=(comp_name,)))
    form = ShipForm()
    return render(request,'comp/comp_enroll_ship.html', {'form': form, 'comp_name': comp_name})

def comp_enroll_plan(request, comp_name):
    if request.method == 'POST':
        form = PlanForm(request.POST)
        print("post")
        print(form.errors)
        print("post")
        if form.is_valid():
            new_item = form.save()
        new_item.save()
        return HttpResponseRedirect(reverse('maritime:comp_main', args=(comp_name,)))
    form = PlanForm()
    return render(request,'comp/comp_enroll_plan.html', {'form': form, 'comp_name': comp_name})


def create(request):
    if request.method == 'POST':
        form = MateForm(request.POST)
        if form.is_valid():
            new_item = form.save()
        return render(request, 'second/create.html', {'form': form})
    form = MateForm()
    return render(request, 'second/create.html', {'form': form})

def mate_checking_rejected(request,comp_name, mate_id):
    Mate_rejected.objects.filter(id=mate_id).delete()
    return HttpResponseRedirect(reverse('maritime:comp_check_rejected',args=(comp_name,)))
def eng_checking_rejected(request,comp_name, eng_id):
    Eng_rejected.objects.filter(id=eng_id).delete()
    return HttpResponseRedirect(reverse('maritime:comp_check_rejected',args=(comp_name,)))
def ship_checking_rejected(request,comp_name, ship_name):
    Ship_rejected.objects.filter(ship_name=ship_name).delete()
    return HttpResponseRedirect(reverse('maritime:comp_check_rejected',args=(comp_name,)))
def plan_checking_rejected(request,comp_name, plan_id):
    Plan_rejected.objects.filter(id=plan_id).delete()
    return HttpResponseRedirect(reverse('maritime:comp_check_rejected',args=(comp_name,)))

def gov_add_new_port(request):
    if request.method == 'POST':
        form = AddNewPortForm(request.POST)
        if form.is_valid():
            new_item = form.save()
            new_item.save()
        return redirect(reverse('maritime:gov_main_page'))
        # return HttpResponseRedirect(reverse('maritime:gov_main_page'))
    form = AddNewPortForm()
    return render(request,'gov/gov_add_new_port.html', {'form': form})


def gov_checking_submitted_mate(request,mate_id):
    mate = Mate_temp.objects.filter(id=mate_id).first()
    new_mate = Mate_stored(mate_name=mate.mate_name, mate_comp=mate.mate_comp, gender=mate.gender, age=mate.age)
    new_mate.save()
    mate.delete()
    return HttpResponseRedirect(reverse('maritime:gov_check_submitted'))
def gov_checking_submitted_eng(request,eng_id):
    eng = Eng_temp.objects.filter(id=eng_id).first()
    new_eng = Eng_stored(eng_name=eng.eng_name, eng_comp=eng.eng_comp, gender=eng.gender, age=eng.age)
    new_eng.save()
    eng.delete()
    return HttpResponseRedirect(reverse('maritime:gov_check_submitted'))
def gov_checking_submitted_ship(request,ship_name):
    ship = Ship_temp.objects.filter(ship_name=ship_name).first()
    new_ship = Ship_stored(ship_name=ship.ship_name, ship_comp=ship.ship_comp, age=ship.age)
    new_ship.save()
    ship.delete()
    return HttpResponseRedirect(reverse('maritime:gov_check_submitted'))
def gov_checking_submitted_plan(request,plan_id):
    plan = Plan_temp.objects.filter(id=plan_id).first()
    new_plan = Plan_stored(plan_port1=plan.plan_port1,plan_port2=plan.plan_port2,departure=plan.departure,arrival=plan.arrival,plan_ship=plan.plan_ship,plan_comp=plan.plan_comp)
    new_plan.save()
    plan.delete()
    return HttpResponseRedirect(reverse('maritime:gov_check_submitted'))
def gov_deny_submitted_mate(request,mate_id):
    mate = Mate_temp.objects.filter(id=mate_id).first()
    new_mate = Mate_rejected(mate_name=mate.mate_name, mate_comp=mate.mate_comp, gender=mate.gender, age=mate.age)
    new_mate.save()
    mate.delete()
    return HttpResponseRedirect(reverse('maritime:gov_check_submitted'))
def gov_deny_submitted_eng(request,eng_id):
    eng = Eng_temp.objects.filter(id=eng_id).first()
    new_eng = Eng_rejected(eng_name=eng.eng_name, eng_comp=eng.eng_comp, gender=eng.gender, age=eng.age)
    new_eng.save()
    eng.delete()
    return HttpResponseRedirect(reverse('maritime:gov_check_submitted'))
def gov_deny_submitted_ship(request,ship_name):
    ship = Ship_temp.objects.filter(ship_name=ship_name).first()
    new_ship = Ship_rejected(ship_name=ship.ship_name, ship_comp=ship.ship_comp, age=ship.age)
    new_ship.save()
    ship.delete()
    return HttpResponseRedirect(reverse('maritime:gov_check_submitted'))
def gov_deny_submitted_plan(request,plan_id):
    plan = Plan_temp.objects.filter(id=plan_id).first()
    new_plan = Plan_rejected(plan_port1=plan.plan_port1,plan_port2=plan.plan_port2,departure=plan.departure,arrival=plan.arrival,plan_ship=plan.plan_ship,plan_comp=plan.plan_comp)
    new_plan.save()
    plan.delete()
    return HttpResponseRedirect(reverse('maritime:gov_check_submitted'))

def comp_check_rejected(request,comp_name):
    comp = get_object_or_404(Comp,pk=comp_name)
    all_mates = Mate_rejected.objects.filter(mate_comp=comp)
    all_engs = Eng_rejected.objects.filter(eng_comp=comp)
    all_ships = Ship_rejected.objects.filter(ship_comp=comp)
    all_plans = Plan_rejected.objects.filter(plan_comp=comp)
    return render(request, 'comp/comp_check_rejected.html', {'comp': comp, 'all_mates': all_mates, 'all_engs': all_engs, 'all_ships': all_ships, 'all_plans': all_plans} )

def comp_check_permitted(request,comp_name):
    comp = get_object_or_404(Comp,pk=comp_name)
    all_mates = Mate_stored.objects.filter(mate_comp=comp)
    all_engs = Eng_stored.objects.filter(eng_comp=comp)
    all_ships = Ship_stored.objects.filter(ship_comp=comp)
    all_plans = Plan_stored.objects.filter(plan_comp=comp)
    return render(request, 'comp/comp_check_permitted.html', {'comp': comp, 'all_mates': all_mates, 'all_engs': all_engs, 'all_ships': all_ships, 'all_plans': all_plans} )

def gov_main(request):
    return HttpResponse("This is the government's main page")
def gov_check_submitted(request):
    all_mates = Mate_temp.objects.all()
    all_engs = Eng_temp.objects.all()
    all_ships = Ship_temp.objects.all()
    all_plans = Plan_temp.objects.all()
    return render(request, 'gov/gov_check_submitted.html', {'all_mates': all_mates, 'all_engs': all_engs, 'all_ships': all_ships, 'all_plans': all_plans} )

def gov_check_permitted(request):
    all_mates = Mate_stored.objects.all()
    all_engs = Eng_stored.objects.all()
    all_ships = Ship_stored.objects.all()
    all_plans = Plan_stored.objects.all()
    all_ports = Port.objects.all()
    return render(request, 'gov/gov_check_permitted.html', {'all_mates': all_mates, 'all_engs': all_engs, 'all_ships': all_ships, 'all_plans': all_plans, 'all_ports': all_ports})

def modify_plan(request,comp_name,plan_id):
    if request.method == 'POST':
        form = ModifyPlanForm(request.POST)
        if form.is_valid():
            plan_rejected = Plan_rejected.objects.get(id=plan_id)
            data = form.cleaned_data
            plan_rejected.departure=data['departure']
            plan_rejected.arrival=data['arrival']
            new_plan = Plan_temp(plan_port1=plan_rejected.plan_port1, plan_port2=plan_rejected.plan_port2, departure=plan_rejected.departure,
                                   arrival=plan_rejected.arrival, plan_ship=plan_rejected.plan_ship, plan_comp=plan_rejected.plan_comp)
            new_plan.save()
            plan_rejected.delete()
        return HttpResponseRedirect(reverse('maritime:comp_main', args=(comp_name,)))
    form = ModifyPlanForm()
    return render(request,'comp/comp_modify_plan.html', {'form': form, 'comp_name': comp_name, 'plan_id': plan_id})
