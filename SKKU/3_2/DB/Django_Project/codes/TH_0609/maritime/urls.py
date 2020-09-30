from django.urls import path, re_path
from . import views

app_name = 'maritime'
urlpatterns=[
    path('maritime_main/', views.maritime_main, name='Maritime_main'),
    path('gov_main_page/',views.gov_main_page, name='gov_main_page'),
    path('comp_main_page/',views.comp_main_page, name='Gov_main_page'),
    path('signup/', views.signup, name='signup'),
    path('comp_signup/', views.signup_comp, name='Comp_signup'),
    path('login/', views.login, name='login'),

    path('connect_to_main/', views.connect_to_main, name='to_main'),
    path('<str:comp_name>/comp_main/', views.comp_main, name='comp_main'),

    path('<str:comp_name>/comp_enroll_select/', views.comp_enroll_select, name='comp_main'),
    path('<str:comp_name>/comp_enroll_mate/', views.comp_enroll_mate, name='comp_enroll_mate'),
    path('<str:comp_name>/comp_enroll_eng/', views.comp_enroll_eng, name='comp_enroll_eng'),
    path('<str:comp_name>/comp_enroll_ship/', views.comp_enroll_ship, name='comp_enroll_ship'),
    path('<str:comp_name>/comp_enroll_plan/', views.comp_enroll_plan, name='comp_enroll_plan'),

    path('<str:comp_name>/comp_check_rejected/', views.comp_check_rejected, name='comp_check_rejected'),

    path('<str:comp_name>/<int:mate_id>/mate_checking_rejected/', views.mate_checking_rejected, name='mate_checking_rejected'),
    path('<str:comp_name>/<int:eng_id>/eng_checking_rejected/', views.eng_checking_rejected, name='eng_checking_rejected'),
    path('<str:comp_name>/<str:ship_name>/ship_checking_rejected/', views.ship_checking_rejected,name='eng_checking_rejected'),
    path('<str:comp_name>/<int:plan_id>/plan_checking_rejected/', views.plan_checking_rejected,name='eng_checking_rejected'),
    path('<str:comp_name>/<int:plan_id>/modify_plan/', views.modify_plan,name='modify_plan'),

    path('gov_check_submitted/', views.gov_check_submitted, name='gov_check_submitted'),
    path('gov_add_new_port/', views.gov_add_new_port, name = 'gov_add_new_port'),

    path('<int:mate_id>/gov_checking_submitted_mate/', views.gov_checking_submitted_mate,name='gov_checking_submitted_mate'),
    path('<int:eng_id>/gov_checking_submitted_eng/', views.gov_checking_submitted_eng,name='gov_checking_rejected_eng'),
    path('<str:ship_name>/gov_checking_submitted_ship/', views.gov_checking_submitted_ship,name='gov_checking_rejected_eng'),
    path('<int:plan_id>/gov_checking_submitted_plan/', views.gov_checking_submitted_plan,name='gov_checking_rejected_eng'),

    path('<int:mate_id>/gov_deny_submitted_mate/', views.gov_deny_submitted_mate,name='gov_deny_submitted_mate'),
    path('<int:eng_id>/gov_deny_submitted_eng/', views.gov_deny_submitted_eng,name='gov_checking_rejected_eng'),
    path('<str:ship_name>/gov_deny_submitted_ship/', views.gov_deny_submitted_ship,name='gov_checking_rejected_eng'),
    path('<int:plan_id>/gov_deny_submitted_plan/', views.gov_deny_submitted_plan, name='gov_checking_rejected_eng'),

    path('<str:comp_name>/comp_check_permitted/', views.comp_check_permitted, name='comp_check_permitted'),
    path('gov_check_permitted/', views.gov_check_permitted, name='gov_check_permitted'),
]
