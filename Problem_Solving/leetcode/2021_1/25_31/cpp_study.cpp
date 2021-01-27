#include <map>
#include <iostream>

using namespace std;

int main() {
    map<string, int> m;

    /*insert*/cout<<"**********insert**********"<<endl;
    /*1*/m.insert(pair<string, int>("marin", 40));
    /*2*/m.insert(make_pair("scv", 60));
    /*3*/m["firebat"] = 50;

    /*iterate*/cout<<"**********iterate**********"<<endl;
    /*1*/map<string, int>::iterator it; // auto it = m.begin()도 가능
    for (it = m.begin(); it != m.end(); it++) {
        cout << it->first << " " << it->second << endl;
    }


    cout<<"TH 1"<<endl;
    for(auto it = m.begin();it!=m.end();it++)
        cout<< it->first << " "<< it->second <<endl;
    cout<<"TH 2"<<endl;
    for(map<string,int>::iterator it = m.begin();it != m.end(); it++)
        cout<< it->first << " "<< it->second <<endl;
    cout<<"TH 3"<<endl;
    for(pair<string, int> atom : m)
    {
        cout<<atom.first<< " "<< atom.second<<endl;
    }

    /*2*/for (pair<string, int> atom : m) {
        cout << atom.first << " " << atom.second << endl;
    }

    /*find*/cout<<"**********find**********"<<endl;
    /*1*/cout<<m.find("scv")->first<<" "<<m.find("scv")->second<<endl;
    /*2*/cout<<"scv"<<" "<<m["scv"]<<endl;

    cout<<"TH 4"<<endl;
    cout<<m.find("scv")->first<<" "<<m.find("scv")->second<<endl;
    cout<<"scv : "<<m["scv"]<<endl;
    

    cout<<"TH//////////////"<<endl;
    map<string,int>::iterator a = m.find("scv");
    cout<<a->first<<endl;
//    m.erase(a);

    /*erase*/cout<<"**********erase**********"<<endl;
    // /*1*/m.erase("scv");
    /*2*/m.erase(m.find("marin"));
    for (pair<string, int> atom : m) {
        cout << atom.first << " " << atom.second << endl;
    }

    /*size*/cout<<"**********size**********"<<endl;
    cout<<m.size()<<endl;

    /*empty*/cout<<"**********empty**********"<<endl;
    cout<<m.empty()<<endl;

    /*size*/cout<<"**********size**********"<<endl;
    cout<<m.size()<<endl;
    return 0;
}