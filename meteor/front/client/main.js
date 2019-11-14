import { Template } from 'meteor/templating';
import { Accounts } from 'meteor/accounts-base';
import './main.html';

Template.body.helpers({
  name: 'Janek',
  surname: 'Smith'
});
