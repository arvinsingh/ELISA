import { Injectable } from '@angular/core';
import { HttpClient, HttpHandler, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})

export class ApiService {

  baseurl = 'http://127.0.0.1:8000/api/';
  httpHeader = new HttpHeaders({'Content-type': 'application/json'});

  constructor(private http: HttpClient) { }

  getAllPlatform(): Observable<any> {
    return this.http.get(this.baseurl + 'platform',
    {headers: this.httpHeader});
  }
}
